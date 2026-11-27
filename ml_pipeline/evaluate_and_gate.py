#!/usr/bin/env python3
"""
Stage 2 of the TinyMLOps pipeline: automated deployment quality gates.

Gates (all thresholds come from config.yaml, CLI flags may override them):
  * QUANT   : the INT8 model has int8 input AND int8 output tensors with a valid
              (scale > 0, -128 <= zero_point <= 127) per-tensor quantization.
  * FLASH   : model_qat_int8.tflite <= gates.max_model_bytes.
  * ACCURACY: INT8 accuracy on all 10,000 MNIST test images >= gates.min_int8_accuracy
              AND (FP32 accuracy - INT8 accuracy) <= gates.max_accuracy_drop.
  * OPS     : every operator in the flatbuffer is in gates.allowed_ops (i.e. is
              registered in the firmware's MicroMutableOpResolver).

If every gate passes, 10 deterministic golden INT8 test vectors (one per digit)
plus their reference INT8 logits are exported for the firmware BIST and the
HIL parity check. If any gate fails the script exits with status 1, which stops
`make`, DVC and the GitHub Actions pipeline before any firmware is generated.

Demo of a blocked deployment:
    python ml_pipeline/evaluate_and_gate.py --max-bytes 10000
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import struct
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "ml_pipeline" / "config.yaml"

# TFLite schema BuiltinOperator enum (tensorflow/lite/schema/schema.fbs).
BUILTIN_OPS = {
    0: "ADD", 1: "AVERAGE_POOL_2D", 2: "CONCATENATION", 3: "CONV_2D", 4: "DEPTHWISE_CONV_2D",
    5: "DEPTH_TO_SPACE", 6: "DEQUANTIZE", 7: "EMBEDDING_LOOKUP", 8: "FLOOR", 9: "FULLY_CONNECTED",
    10: "HASHTABLE_LOOKUP", 11: "L2_NORMALIZATION", 12: "L2_POOL_2D",
    13: "LOCAL_RESPONSE_NORMALIZATION", 14: "LOGISTIC", 15: "LSH_PROJECTION", 16: "LSTM",
    17: "MAX_POOL_2D", 18: "MUL", 19: "RELU", 20: "RELU_N1_TO_1", 21: "RELU6", 22: "RESHAPE",
    23: "RESIZE_BILINEAR", 24: "RNN", 25: "SOFTMAX", 26: "SPACE_TO_DEPTH", 27: "SVDF", 28: "TANH",
    29: "CONCAT_EMBEDDINGS", 30: "SKIP_GRAM", 31: "CALL", 32: "CUSTOM",
    33: "EMBEDDING_LOOKUP_SPARSE", 34: "PAD", 35: "UNIDIRECTIONAL_SEQUENCE_RNN", 36: "GATHER",
    37: "BATCH_TO_SPACE_ND", 38: "SPACE_TO_BATCH_ND", 39: "TRANSPOSE", 40: "MEAN", 41: "SUB",
    42: "DIV", 43: "SQUEEZE", 44: "UNIDIRECTIONAL_SEQUENCE_LSTM", 45: "STRIDED_SLICE",
    46: "BIDIRECTIONAL_SEQUENCE_RNN", 47: "EXP", 48: "TOPK_V2", 49: "SPLIT", 50: "LOG_SOFTMAX",
    51: "DELEGATE", 52: "BIDIRECTIONAL_SEQUENCE_LSTM", 53: "CAST", 54: "PRELU", 55: "MAXIMUM",
    56: "ARG_MAX", 57: "MINIMUM", 58: "LESS", 59: "NEG", 60: "PADV2", 61: "GREATER",
    62: "GREATER_EQUAL", 63: "LESS_EQUAL", 64: "SELECT", 65: "SLICE", 66: "SIN",
    67: "TRANSPOSE_CONV", 68: "SPARSE_TO_DENSE", 69: "TILE", 70: "EXPAND_DIMS", 71: "EQUAL",
    72: "NOT_EQUAL", 73: "LOG", 74: "SUM", 75: "SQRT", 76: "RSQRT", 77: "SHAPE", 78: "POW",
    79: "ARG_MIN", 80: "FAKE_QUANT", 81: "REDUCE_PROD", 82: "REDUCE_MAX", 83: "PACK",
    84: "LOGICAL_OR", 85: "ONE_HOT", 86: "LOGICAL_AND", 87: "LOGICAL_NOT", 88: "UNPACK",
    89: "REDUCE_MIN", 90: "FLOOR_DIV", 91: "REDUCE_ANY", 92: "SQUARE", 93: "ZEROS_LIKE",
    94: "FILL", 95: "FLOOR_MOD", 96: "RANGE", 97: "RESIZE_NEAREST_NEIGHBOR", 98: "LEAKY_RELU",
    99: "SQUARED_DIFFERENCE", 100: "MIRROR_PAD", 101: "ABS", 102: "SPLIT_V", 103: "UNIQUE",
    104: "CEIL", 105: "REVERSE_V2", 106: "ADD_N", 107: "GATHER_ND", 108: "COS", 109: "WHERE",
    110: "RANK", 111: "ELU", 112: "REVERSE_SEQUENCE", 113: "MATRIX_DIAG", 114: "QUANTIZE",
    115: "MATRIX_SET_DIAG", 116: "ROUND", 117: "HARD_SWISH", 118: "IF", 119: "WHILE",
    120: "NON_MAX_SUPPRESSION_V4", 121: "NON_MAX_SUPPRESSION_V5", 122: "SCATTER_ND",
    123: "SELECT_V2", 124: "DENSIFY", 125: "SEGMENT_SUM", 126: "BATCH_MATMUL",
}
CUSTOM_OP_CODE = 32


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def load_config(path: str | Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass
class GateResult:
    name: str
    passed: bool
    measured: str
    threshold: str
    detail: str = ""


# --------------------------------------------------------------------------- #
# Dependency-free TFLite flatbuffer operator audit
# --------------------------------------------------------------------------- #
class _FB:
    """Minimal read-only FlatBuffers accessor (enough to walk Model.operator_codes)."""

    def __init__(self, buf: bytes):
        self.buf = buf

    def u8(self, o: int) -> int: return self.buf[o]
    def i8(self, o: int) -> int: return struct.unpack_from("<b", self.buf, o)[0]
    def u16(self, o: int) -> int: return struct.unpack_from("<H", self.buf, o)[0]
    def i32(self, o: int) -> int: return struct.unpack_from("<i", self.buf, o)[0]
    def u32(self, o: int) -> int: return struct.unpack_from("<I", self.buf, o)[0]

    def field_pos(self, table: int, field_index: int) -> Optional[int]:
        """Absolute position of a field's data, or None when the field is absent (default)."""
        vtable = table - self.i32(table)
        vt_size = self.u16(vtable)
        entry = 4 + 2 * field_index
        if entry >= vt_size:
            return None
        rel = self.u16(vtable + entry)
        return table + rel if rel else None

    def deref(self, pos: int) -> int:
        return pos + self.u32(pos)

    def vector(self, table: int, field_index: int) -> List[int]:
        """Positions of table elements of a vector-of-tables field."""
        pos = self.field_pos(table, field_index)
        if pos is None:
            return []
        vec = self.deref(pos)
        n = self.u32(vec)
        return [self.deref(vec + 4 + 4 * i) for i in range(n)]

    def string(self, table: int, field_index: int) -> Optional[str]:
        pos = self.field_pos(table, field_index)
        if pos is None:
            return None
        s = self.deref(pos)
        n = self.u32(s)
        return self.buf[s + 4 : s + 4 + n].decode("utf-8", errors="replace")


def parse_tflite_opcodes(model_bytes: bytes) -> List[str]:
    """Return the operator names referenced by a .tflite flatbuffer (in opcode-table order).

    Model.operator_codes is field #1; OperatorCode has deprecated_builtin_code (int8, #0),
    custom_code (string, #1), version (#2) and builtin_code (int32, #3). The real code is
    max(deprecated_builtin_code, builtin_code) per the schema's compatibility rule.
    """
    if len(model_bytes) < 8 or model_bytes[4:8] != b"TFL3":
        raise ValueError("not a TFLite flatbuffer (missing 'TFL3' file identifier)")
    fb = _FB(model_bytes)
    root = fb.u32(0)
    names: List[str] = []
    for oc in fb.vector(root, 1):
        p_dep = fb.field_pos(oc, 0)
        dep = fb.i8(p_dep) if p_dep is not None else 0
        p_new = fb.field_pos(oc, 3)
        new = fb.i32(p_new) if p_new is not None else 0
        code = max(dep, new)
        if code == CUSTOM_OP_CODE:
            names.append(f"CUSTOM:{fb.string(oc, 1) or '?'}")
        else:
            names.append(BUILTIN_OPS.get(code, f"BUILTIN_{code}"))
    return names


# --------------------------------------------------------------------------- #
# Pure gate logic (unit-tested without TensorFlow)
# --------------------------------------------------------------------------- #
def check_size_gate(model_size: int, max_bytes: int) -> GateResult:
    return GateResult("FLASH_SIZE", model_size <= max_bytes, f"{model_size:,} B", f"<= {max_bytes:,} B",
                      f"{model_size / 1024:.2f} KiB = {100 * model_size / 4194304:.3f}% of 4 MB flash")


def check_accuracy_gate(int8_acc: float, fp32_acc: float, min_acc: float, max_drop: float) -> List[GateResult]:
    drop = fp32_acc - int8_acc
    return [
        GateResult("INT8_ACCURACY", int8_acc >= min_acc, f"{100 * int8_acc:.2f}%", f">= {100 * min_acc:.2f}%"),
        GateResult("ACCURACY_DROP", drop <= max_drop, f"{100 * drop:+.2f} pp", f"<= {100 * max_drop:.2f} pp",
                   f"FP32 {100 * fp32_acc:.2f}% -> INT8 {100 * int8_acc:.2f}%"),
    ]


def check_operator_gate(ops: Sequence[str], allowed: Sequence[str]) -> GateResult:
    bad = sorted(set(ops) - set(allowed))
    return GateResult("OPERATOR_AUDIT", not bad, ", ".join(ops) or "(none)", "subset of " + ", ".join(allowed),
                      ("unsupported: " + ", ".join(bad)) if bad else "all ops registered in MicroMutableOpResolver<6>")


def check_quantization_gate(input_detail: Dict[str, Any], output_detail: Dict[str, Any]) -> GateResult:
    problems = []
    for role, d in (("input", input_detail), ("output", output_detail)):
        dtype = np.dtype(d["dtype"]).name
        scale, zp = d["quantization"]
        if dtype != "int8":
            problems.append(f"{role} dtype={dtype}")
        if not scale or scale <= 0:
            problems.append(f"{role} scale={scale}")
        if not -128 <= int(zp) <= 127:
            problems.append(f"{role} zero_point={zp}")
    i_s, i_z = input_detail["quantization"]
    o_s, o_z = output_detail["quantization"]
    measured = f"in int8 s={i_s:.6g} zp={i_z} | out int8 s={o_s:.6g} zp={o_z}"
    return GateResult("INT8_IO_QUANT", not problems, measured, "int8 in/out, scale>0, zp in [-128,127]",
                      "; ".join(problems))


def quantize(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return np.clip(np.round(x / scale) + zero_point, -128, 127).astype(np.int8)


def dequantize(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (q.astype(np.float32) - zero_point) * scale


# --------------------------------------------------------------------------- #
# TFLite evaluation (TensorFlow imported lazily)
# --------------------------------------------------------------------------- #
def make_interpreter(model_path: Path, reference_kernels: bool = True):
    """Reference kernels = same integer arithmetic as TFLite Micro -> bit-exact parity."""
    import tensorflow as tf

    kwargs = {}
    if reference_kernels:
        kwargs["experimental_op_resolver_type"] = tf.lite.experimental.OpResolverType.BUILTIN_REF
    interp = tf.lite.Interpreter(model_path=str(model_path), **kwargs)
    interp.allocate_tensors()
    return interp


def run_int8(interp, q_images: np.ndarray) -> np.ndarray:
    """Run int8 images (N, 28, 28, 1) -> int8 logits (N, 10)."""
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    res = np.empty((len(q_images), out["shape"][-1]), dtype=np.int8)
    for i, img in enumerate(q_images):
        interp.set_tensor(inp["index"], img[np.newaxis].astype(np.int8))
        interp.invoke()
        res[i] = interp.get_tensor(out["index"])[0]
    return res


def run_fp32(interp, images: np.ndarray) -> np.ndarray:
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    res = np.empty((len(images), out["shape"][-1]), dtype=np.float32)
    for i, img in enumerate(images):
        interp.set_tensor(inp["index"], img[np.newaxis].astype(np.float32))
        interp.invoke()
        res[i] = interp.get_tensor(out["index"])[0]
    return res


def load_mnist_test() -> tuple[np.ndarray, np.ndarray]:
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    import tensorflow as tf

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_test.astype(np.float32) / 255.0)[..., np.newaxis], y_test.astype(np.int64)


def select_golden_vectors(q_images: np.ndarray, labels: np.ndarray, preds: np.ndarray,
                          num_classes: int = 10) -> List[int]:
    """Deterministic: the first test-set index of each digit that the INT8 model classifies correctly."""
    chosen = []
    for digit in range(num_classes):
        hits = np.where((labels == digit) & (preds == digit))[0]
        if len(hits) == 0:
            raise RuntimeError(f"INT8 model never classifies digit {digit} correctly")
        chosen.append(int(hits[0]))
    return chosen


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_table(results: List[GateResult]) -> None:
    w = (16, 6, 46, 40)
    line = "+" + "+".join("-" * (x + 2) for x in w) + "+"
    print(line)
    print("| " + " | ".join(h.ljust(x) for h, x in zip(("GATE", "STATUS", "MEASURED", "THRESHOLD"), w)) + " |")
    print(line)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print("| " + " | ".join(str(v)[:x].ljust(x) for v, x in zip((r.name, status, r.measured, r.threshold), w)) + " |")
        if r.detail:
            print("| " + " " * w[0] + " | " + " " * w[1] + " | " + ("  " + r.detail)[: w[2] + w[3] + 3].ljust(w[2] + w[3] + 3) + " |")
    print(line)


def log_gate_to_mlflow(cfg: Dict[str, Any], report: Dict[str, Any]) -> None:
    metrics_path = resolve(cfg["paths"]["metrics_json"])
    try:
        import mlflow

        run_id = json.loads(metrics_path.read_text())["mlflow_run_id"]
        if not run_id:
            return
        mlflow.set_tracking_uri(resolve(cfg["paths"]["mlflow_tracking_uri"]).as_uri())
        with mlflow.start_run(run_id=run_id):
            m = report["metrics"]
            mlflow.log_metrics({k: float(v) for k, v in m.items() if isinstance(v, (int, float))})
            mlflow.set_tag("deployment_gate", "PASSED" if report["passed"] else "BLOCKED")
    except Exception as exc:  # tracking must never break the gate itself
        print(f"[mlflow] gate metrics not logged ({exc.__class__.__name__}: {exc})")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--max-bytes", type=int, help="override gates.max_model_bytes (Flash ceiling)")
    ap.add_argument("--min-accuracy", type=float, help="override gates.min_int8_accuracy (0..1)")
    ap.add_argument("--max-drop", type=float, help="override gates.max_accuracy_drop (0..1)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N test images (fast smoke test)")
    ap.add_argument("--fp32-model", help="override path to baseline FP32 .tflite")
    ap.add_argument("--int8-model", help="override path to INT8 .tflite")
    ap.add_argument("--report-out", help="override gate report JSON path")
    ap.add_argument("--golden-out", help="override golden vectors JSON path")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    g = cfg["gates"]
    max_bytes = args.max_bytes if args.max_bytes is not None else int(g["max_model_bytes"])
    min_acc = args.min_accuracy if args.min_accuracy is not None else float(g["min_int8_accuracy"])
    max_drop = args.max_drop if args.max_drop is not None else float(g["max_accuracy_drop"])

    fp32_path = Path(args.fp32_model) if args.fp32_model else resolve(cfg["paths"]["baseline_tflite"])
    int8_path = Path(args.int8_model) if args.int8_model else resolve(cfg["paths"]["int8_tflite"])
    report_path = Path(args.report_out) if args.report_out else resolve(cfg["paths"]["gate_report_json"])
    golden_path = Path(args.golden_out) if args.golden_out else resolve(cfg["paths"]["golden_vectors_json"])

    print("=" * 72)
    print(" TinyMLOps | Stage 2: deployment quality gates")
    print("=" * 72)
    for p in (fp32_path, int8_path):
        if not p.exists():
            print(f"[error] missing {p} - run `make train` first")
            return 1

    int8_bytes = int8_path.read_bytes()
    results: List[GateResult] = []

    # 1) quantization params
    q_interp = make_interpreter(int8_path, reference_kernels=True)
    in_d, out_d = q_interp.get_input_details()[0], q_interp.get_output_details()[0]
    results.append(check_quantization_gate(in_d, out_d))
    in_scale, in_zp = float(in_d["quantization"][0]), int(in_d["quantization"][1])
    out_scale, out_zp = float(out_d["quantization"][0]), int(out_d["quantization"][1])

    # 2) flash size
    results.append(check_size_gate(len(int8_bytes), max_bytes))

    # 3) operator audit
    ops = parse_tflite_opcodes(int8_bytes)
    results.append(check_operator_gate(ops, g["allowed_ops"]))

    # 4) accuracy (full 10k test set unless --limit)
    x_test, y_test = load_mnist_test()
    if args.limit:
        x_test, y_test = x_test[: args.limit], y_test[: args.limit]
    t0 = time.time()
    fp32_logits = run_fp32(make_interpreter(fp32_path, reference_kernels=False), x_test)
    q_images = quantize(x_test, in_scale, in_zp)
    int8_logits = run_int8(q_interp, q_images)
    eval_s = time.time() - t0
    fp32_pred = fp32_logits.argmax(1)
    int8_pred = int8_logits.argmax(1)
    fp32_acc = float((fp32_pred == y_test).mean())
    int8_acc = float((int8_pred == y_test).mean())
    agreement = float((fp32_pred == int8_pred).mean())
    print(f"[eval] {len(y_test):,} images in {eval_s:.1f}s | FP32={100 * fp32_acc:.2f}% "
          f"INT8={100 * int8_acc:.2f}% | FP32/INT8 top-1 agreement={100 * agreement:.2f}%")
    results.extend(check_accuracy_gate(int8_acc, fp32_acc, min_acc, max_drop))

    passed = all(r.passed for r in results)
    print_table(results)

    report = {
        "passed": passed,
        "evaluated_images": int(len(y_test)),
        "operators": ops,
        "quantization": {"input_scale": in_scale, "input_zero_point": in_zp,
                         "output_scale": out_scale, "output_zero_point": out_zp},
        "thresholds": {"max_model_bytes": max_bytes, "min_int8_accuracy": min_acc, "max_accuracy_drop": max_drop},
        "metrics": {
            "fp32_tflite_accuracy": fp32_acc,
            "int8_tflite_accuracy": int8_acc,
            "accuracy_drop": fp32_acc - int8_acc,
            "fp32_int8_agreement": agreement,
            "int8_model_bytes": len(int8_bytes),
            "fp32_model_bytes": fp32_path.stat().st_size,
        },
        "gates": [asdict(r) for r in results],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if passed:
        idx = select_golden_vectors(q_images, y_test, int8_pred)
        golden = {
            "selection": "first_correct_per_class",
            "input_scale": in_scale, "input_zero_point": in_zp,
            "output_scale": out_scale, "output_zero_point": out_zp,
            "model_sha256": __import__("hashlib").sha256(int8_bytes).hexdigest(),
            "vectors": [
                {
                    "test_index": i,
                    "label": int(y_test[i]),
                    "image_int8": q_images[i].reshape(-1).astype(int).tolist(),
                    "expected_logits_int8": int8_logits[i].astype(int).tolist(),
                    "expected_pred": int(int8_pred[i]),
                }
                for i in idx
            ],
        }
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(golden) + "\n", encoding="utf-8")
        print(f"[golden] exported 10 vectors (test indices {idx}) -> {golden_path}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.no_mlflow and args.report_out is None:
        log_gate_to_mlflow(cfg, report)

    if passed:
        print("[GATE] ALL QUALITY GATES PASSED -> model approved for ESP32 deployment")
        return 0
    failed = ", ".join(r.name for r in results if not r.passed)
    print(f"[GATE] DEPLOYMENT BLOCKED -> failed gate(s): {failed}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
