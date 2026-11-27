#!/usr/bin/env python3
"""
Layer 2 / Layer 3 verifier: host <-> ESP32 bit-exact parity, accuracy and latency.

Modes
-----
1) Live device (real ESP32 on USB, or the Wokwi VS Code simulator's RFC2217 port):

     python tests/serial_hil_verifier.py --port COM5
     python tests/serial_hil_verifier.py --port /dev/ttyUSB0 --extra 200
     python tests/serial_hil_verifier.py --port rfc2217://localhost:4000

   Streams every golden vector (and optionally N extra MNIST test images) as
   `INFER:<784 int8>` frames, parses the JSON reply and compares the 10 int8 logits
   BIT-FOR-BIT against the Python tf.lite.Interpreter (reference kernels, the same
   integer arithmetic as TFLite Micro). Reports device accuracy and latency.

2) Host-only reference check (no hardware; runs in CI):

     python tests/serial_hil_verifier.py --host-only

   Re-runs the TFLite interpreter on the golden vectors and checks that its logits
   equal the ones embedded in firmware_esp32/src/test_vectors.h (i.e. the firmware's
   on-device BIST compares against a valid reference).

3) Simulation log parser (CI, after wokwi-cli):

     python tests/serial_hil_verifier.py --parse-log sim_serial.log --summary-md report.md

   Parses the BIST table / summary lines, writes a Markdown report (for
   $GITHUB_STEP_SUMMARY) and exits non-zero on FATAL, Guru Meditation, or < 10/10.

Exit code 0 = verified, 1 = verification failed, 2 = usage / connection error.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "ml_pipeline" / "config.yaml"
VECTORS_H = REPO_ROOT / "firmware_esp32" / "src" / "test_vectors.h"

SUMMARY_RE = re.compile(
    r"\[TINYMLOPS_TEST_SUMMARY\]\s+PASSED=(\d+)/(\d+)\s+ACCURACY=([\d.]+)%\s+"
    r"AVG_LATENCY_US=([\d.]+)\s+ARENA_BYTES=(\d+)")
ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)%\s*\|\s*(\d+)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|")
KV_RE = re.compile(r"([A-Z0-9_]+)\s*=\s*([^\s]+)")


def load_cfg() -> Dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def golden_path(cfg) -> Path:
    return REPO_ROOT / cfg["paths"]["golden_vectors_json"]


def model_path(cfg) -> Path:
    return REPO_ROOT / cfg["paths"]["int8_tflite"]


# --------------------------------------------------------------------------- #
# Host reference
# --------------------------------------------------------------------------- #
class HostReference:
    def __init__(self, tflite_path: Path):
        import tensorflow as tf

        self.interp = tf.lite.Interpreter(
            model_path=str(tflite_path),
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.in_scale, self.in_zp = self.inp["quantization"]

    def logits(self, img_int8: np.ndarray) -> np.ndarray:
        self.interp.set_tensor(self.inp["index"], img_int8.reshape(1, 28, 28, 1).astype(np.int8))
        self.interp.invoke()
        return self.interp.get_tensor(self.out["index"])[0].astype(int)

    def quantize(self, x_float: np.ndarray) -> np.ndarray:
        return np.clip(np.round(x_float / self.in_scale) + self.in_zp, -128, 127).astype(np.int8)


def parse_vectors_header(path: Path = VECTORS_H) -> Dict[str, Any]:
    """Extract labels / expected logits / images from the generated C++ header."""
    text = path.read_text(encoding="utf-8")

    def block(name: str) -> List[int]:
        m = re.search(rf"{name}\[[^=]*=\s*\{{(.*?)\}};", text, re.S)
        if not m:
            raise ValueError(f"{name} not found in {path}")
        body = re.sub(r"//[^\n]*", "", m.group(1))
        return [int(v) for v in re.findall(r"-?\d+", body)]

    labels = block("kTestLabels")
    logits = np.array(block("kExpectedLogits")).reshape(10, 10)
    images = np.array(block("kTestImages")).reshape(10, 784)
    return {"labels": labels, "logits": logits, "images": images}


def host_only(cfg) -> int:
    ref = HostReference(model_path(cfg))
    hdr = parse_vectors_header()
    gold = json.loads(golden_path(cfg).read_text())
    ok = True
    print("+-------+-------+---------------------------------------------+--------+")
    print("| Digit | Label | Host TFLite logits (int8)                   | Parity |")
    print("+-------+-------+---------------------------------------------+--------+")
    for i in range(10):
        host = ref.logits(hdr["images"][i])
        exact = np.array_equal(host, hdr["logits"][i]) and \
            np.array_equal(hdr["images"][i], np.array(gold["vectors"][i]["image_int8"]))
        ok &= bool(exact) and int(host.argmax()) == hdr["labels"][i]
        print(f"| {i:5d} | {hdr['labels'][i]:5d} | {' '.join(f'{v:4d}' for v in host):43s} | "
              f"{'EXACT' if exact else 'DIFF':6s} |")
    print("+-------+-------+---------------------------------------------+--------+")
    print(f"[HOST_PARITY] {'PASS' if ok else 'FAIL'} - test_vectors.h reference logits "
          f"{'match' if ok else 'DO NOT match'} tf.lite.Interpreter (BUILTIN_REF)")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Live serial device
# --------------------------------------------------------------------------- #
def open_port(url: str, baud: int, timeout: float):
    import serial

    if "://" in url:
        return serial.serial_for_url(url, baudrate=baud, timeout=timeout)
    return serial.Serial(url, baudrate=baud, timeout=timeout)


def read_until(ser, predicate, timeout: float, echo: bool = False) -> Optional[str]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        if echo and line:
            print(f"   esp32> {line}")
        if predicate(line):
            return line
    return None


def device_infer(ser, img: np.ndarray, timeout: float) -> Dict[str, Any]:
    ser.write(("INFER:" + ",".join(str(int(v)) for v in img.reshape(-1)) + "\n").encode())
    ser.flush()
    line = read_until(ser, lambda l: l.startswith("{"), timeout)
    if line is None:
        raise TimeoutError("no JSON reply from device")
    reply = json.loads(line)
    if "error" in reply:
        raise RuntimeError(f"device error: {reply['error']}")
    return reply


def live(cfg, args) -> int:
    ref = HostReference(model_path(cfg))
    gold = json.loads(golden_path(cfg).read_text())
    try:
        ser = open_port(args.port, args.baud, timeout=0.5)
    except Exception as exc:
        print(f"[error] cannot open {args.port}: {exc}")
        return 2
    with ser:
        if args.reset:
            ser.dtr = False
            ser.rts = True
            time.sleep(0.1)
            ser.rts = False
        print(f"[hil] connected to {args.port} @ {args.baud} baud - waiting for firmware ...")
        if args.wait_boot:
            summary = read_until(ser, lambda l: "[TINYMLOPS_READY]" in l or "[TINYMLOPS_FATAL]" in l,
                                 args.boot_timeout, echo=True)
            if summary is None or "FATAL" in summary:
                print("[error] firmware did not become ready")
                return 1
        ser.reset_input_buffer()
        ser.write(b"PING\n")
        if read_until(ser, lambda l: l == "PONG", 5.0) is None:
            print("[error] device did not answer PING")
            return 2

        samples = [(f"golden[{i}]", np.array(v["image_int8"], dtype=np.int8), v["label"])
                   for i, v in enumerate(gold["vectors"])]
        if args.extra:
            os.environ["TF_USE_LEGACY_KERAS"] = "1"
            import tensorflow as tf

            (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            rng = np.random.default_rng(args.seed)
            for i in rng.choice(len(x_test), size=args.extra, replace=False):
                img = ref.quantize(x_test[i].astype(np.float32) / 255.0)
                samples.append((f"test[{i}]", img, int(y_test[i])))

        exact = correct = 0
        lat: List[int] = []
        print("+--------------+-------+------+------+-------------+--------+")
        print("| Sample       | Label | Host | ESP  | Latency(us) | Parity |")
        print("+--------------+-------+------+------+-------------+--------+")
        for name, img, label in samples:
            host = ref.logits(img)
            dev = device_infer(ser, img, args.timeout)
            dev_logits = np.array(dev["logits"], dtype=int)
            same = np.array_equal(host, dev_logits)
            exact += same
            correct += dev["pred"] == label
            lat.append(int(dev["latency_us"]))
            print(f"| {name:12s} | {label:5d} | {int(host.argmax()):4d} | {dev['pred']:4d} | "
                  f"{dev['latency_us']:11d} | {'EXACT' if same else 'DIFF':6s} |")
            if not same and args.verbose:
                print(f"     host={host.tolist()}\n     esp ={dev_logits.tolist()}")
        print("+--------------+-------+------+------+-------------+--------+")

    n = len(samples)
    budget = int(cfg["hardware"]["max_avg_latency_us"])
    avg = statistics.mean(lat)
    print(f"[HIL_SUMMARY] SAMPLES={n} BITEXACT={exact}/{n} DEVICE_ACCURACY={100 * correct / n:.2f}% "
          f"AVG_LATENCY_US={avg:.1f} P95_LATENCY_US={np.percentile(lat, 95):.0f} BUDGET_US={budget}")
    ok = exact == n and avg <= budget
    print(f"[HIL] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Simulation / device log parser
# --------------------------------------------------------------------------- #
def parse_log(text: str) -> Dict[str, Any]:
    res: Dict[str, Any] = {"rows": [], "telemetry": {}, "fatal": [], "summary": None}
    for line in text.splitlines():
        line = line.strip()
        if "[TINYMLOPS_FATAL]" in line or "Guru Meditation" in line:
            res["fatal"].append(line)
        if line.startswith("[TELEMETRY]"):
            res["telemetry"].update(dict(KV_RE.findall(line)))
        if line.startswith("[TINYMLOPS_LATENCY]") or line.startswith("[TINYMLOPS_PARITY]"):
            res["telemetry"].update(dict(KV_RE.findall(line)))
        m = SUMMARY_RE.search(line)
        if m:
            res["summary"] = {"passed": int(m[1]), "total": int(m[2]), "accuracy": float(m[3]),
                              "avg_latency_us": float(m[4]), "arena_bytes": int(m[5])}
        r = ROW_RE.match(line)
        if r:
            res["rows"].append({"digit": int(r[1]), "expected": int(r[2]), "predicted": int(r[3]),
                                "confidence": float(r[4]), "latency_us": int(r[5]),
                                "parity": r[6], "status": r[7]})
    return res


def render_markdown(res: Dict[str, Any], cfg: Dict[str, Any], ok: bool) -> str:
    t, s = res["telemetry"], res["summary"] or {}
    hw = cfg["hardware"]
    arena = s.get("arena_bytes", 0)
    out = [
        "## TinyMLOps Deployment & Simulation Report (Wokwi ESP32-WROOM-32)",
        "",
        f"**Result:** {'PASSED' if ok else 'FAILED'}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| BIST | {s.get('passed', 0)}/{s.get('total', 10)} digits ({s.get('accuracy', 0):.2f}%) |",
        f"| Bit-exact parity vs. host TFLite | {t.get('BITEXACT', 'n/a')} |",
        f"| Avg / min / max latency | {s.get('avg_latency_us', 0):.1f} / {t.get('MIN_US', 'n/a')} / {t.get('MAX_US', 'n/a')} us |",
        f"| Tensor arena used | {arena:,} / {hw['tensor_arena_bytes']:,} B ({100 * arena / hw['tensor_arena_bytes']:.1f}%) |",
        f"| Model size / SHA-256 | {t.get('MODEL_BYTES', 'n/a')} B / `{t.get('MODEL_SHA256_EMBEDDED', 'n/a')[:16]}...` |",
        f"| Model integrity (on-device SHA-256) | {t.get('MODEL_INTEGRITY', 'n/a')} |",
        f"| Free heap before / after AllocateTensors | {t.get('HEAP_FREE_BEFORE_ALLOC', 'n/a')} / {t.get('HEAP_FREE_AFTER_ALLOC', 'n/a')} B |",
        f"| Firmware | {t.get('FW_VERSION', 'n/a')} on {t.get('CHIP', 'ESP32')} @ {t.get('CPU_MHZ', '240')} MHz |",
        "",
    ]
    if res["rows"]:
        out += ["| Digit | Expected | Predicted | Confidence | Latency (us) | Parity | Status |",
                "|---|---|---|---|---|---|---|"]
        out += [f"| {r['digit']} | {r['expected']} | {r['predicted']} | {r['confidence']:.2f}% | "
                f"{r['latency_us']} | {r['parity']} | {'PASS' if r['status'] == 'PASS' else 'FAIL'} |"
                for r in res["rows"]]
    if res["fatal"]:
        out += ["", "### Fatal errors", "```"] + res["fatal"] + ["```"]
    return "\n".join(out) + "\n"


def parse_log_mode(cfg, args) -> int:
    path = Path(args.parse_log)
    if not path.exists():
        print(f"[error] log file {path} not found")
        return 2
    res = parse_log(path.read_text(encoding="utf-8", errors="replace"))
    s = res["summary"]
    ok = (not res["fatal"] and s is not None and s["passed"] == s["total"] == 10
          and s["avg_latency_us"] <= cfg["hardware"]["max_avg_latency_us"]
          and s["arena_bytes"] <= cfg["hardware"]["tensor_arena_bytes"])
    md = render_markdown(res, cfg, ok)
    print(md)
    if args.summary_md:
        Path(args.summary_md).write_text(md, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0 if ok else 1


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--port", help="serial port (COM5, /dev/ttyUSB0) or URL (rfc2217://localhost:4000)")
    mode.add_argument("--host-only", action="store_true", help="host reference parity check, no hardware")
    mode.add_argument("--parse-log", metavar="LOG", help="parse a captured serial/Wokwi log")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=10.0, help="per-inference reply timeout (s)")
    ap.add_argument("--extra", type=int, default=0, help="also stream N random MNIST test images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wait-boot", action="store_true", help="wait for [TINYMLOPS_READY] first")
    ap.add_argument("--boot-timeout", type=float, default=60.0)
    ap.add_argument("--reset", action="store_true", help="toggle RTS to reset a real dev board")
    ap.add_argument("--summary-md", help="(parse-log) write Markdown report here")
    ap.add_argument("--json-out", help="(parse-log) write parsed JSON here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_cfg()
    if args.parse_log:
        return parse_log_mode(cfg, args)
    if args.host_only:
        return host_only(cfg)
    return live(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
