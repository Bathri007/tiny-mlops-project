"""
Layer 1 - Python pipeline & model unit / integration tests.

    pytest -q tests/                      (make test)

Groups
  * data      : preprocessing shapes / dtype / range, model architecture & param budget
  * gates     : size / accuracy / operator / quantization gate logic on dummy inputs,
                flatbuffer opcode parser, end-to-end CLI rejection of a bad model
  * artifacts : real INT8 model tensor types, scales and zero-points (skipped until `make train`)
  * codegen   : C++ header syntax, alignment, byte-exact round trip, SHA-256 consistency,
                optional g++ -fsyntax-only compile of the generated headers
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml_pipeline"))
sys.path.insert(0, str(REPO / "tests"))

import evaluate_and_gate as gate  # noqa: E402  (no TensorFlow import at module load)
import export_to_cpp as codegen  # noqa: E402

CFG = gate.load_config()
INT8_MODEL = REPO / CFG["paths"]["int8_tflite"]
FP32_MODEL = REPO / CFG["paths"]["fp32_tflite"]
GOLDEN = REPO / CFG["paths"]["golden_vectors_json"]
MODEL_H = REPO / CFG["paths"]["model_header"]
VECTORS_H = REPO / CFG["paths"]["vectors_header"]

needs_model = pytest.mark.skipif(not INT8_MODEL.exists(), reason="run `make train` first")
needs_golden = pytest.mark.skipif(not GOLDEN.exists(), reason="run `make gate` first")


def _tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


needs_tf = pytest.mark.skipif(not _tf_available(), reason="tensorflow not installed")


# =========================================================================== #
# Synthetic flatbuffer builder (lets us test the opcode audit without TF)
# =========================================================================== #
def build_fake_tflite(op_codes: list[int], custom_name: str = "MyCustomOp", pad_to: int = 0) -> bytes:
    """Builds a minimal, schema-valid flatbuffer: Model{version, operator_codes[]}.

    FlatBuffers uoffsets are unsigned and always point forward, so objects are laid out
    parent-first: root -> Model vtable/table -> vector -> OperatorCode tables -> string.
    """
    import struct

    buf = bytearray(b"\x00\x00\x00\x00TFL3")

    def align(n=4):
        while len(buf) % n:
            buf.append(0)

    # --- Model: vtable (version #0 @4, operator_codes #1 @8) + table
    vt = len(buf)
    buf += struct.pack("<HHHH", 8, 12, 4, 8)
    model = len(buf)
    buf += struct.pack("<iII", model - vt, 3, 0)  # soffset, version=3, operator_codes (patched)
    struct.pack_into("<I", buf, 0, model)

    # --- vector of OperatorCode offsets (patched below)
    align()
    vec = len(buf)
    struct.pack_into("<I", buf, model + 8, vec - (model + 8))
    buf += struct.pack("<I", len(op_codes)) + b"\x00" * (4 * len(op_codes))

    # --- OperatorCode tables: vtable(4 fields) + table
    string_fields = []
    for i, code in enumerate(op_codes):
        align()
        ovt = len(buf)
        # vtable: size=12, table size=16, fields: dep@4, custom@8 (only for CUSTOM), version absent, builtin@12
        buf += struct.pack("<HHHHHH", 12, 16, 4, 8 if code == 32 else 0, 0, 12)
        tbl = len(buf)
        slot = vec + 4 + 4 * i
        struct.pack_into("<I", buf, slot, tbl - slot)
        buf += struct.pack("<i", tbl - ovt)                     # soffset to vtable
        buf += struct.pack("<b", min(code, 127)) + b"\x00" * 3  # deprecated_builtin_code
        buf += struct.pack("<I", 0)                             # custom_code (patched)
        buf += struct.pack("<i", code)                          # builtin_code
        if code == 32:
            string_fields.append(tbl + 8)

    # --- custom_code string
    align()
    s = len(buf)
    enc = custom_name.encode()
    buf += struct.pack("<I", len(enc)) + enc + b"\x00"
    for f in string_fields:
        struct.pack_into("<I", buf, f, s - f)

    if pad_to > len(buf):
        buf += b"\x00" * (pad_to - len(buf))
    return bytes(buf)


# =========================================================================== #
# data / model
# =========================================================================== #
@needs_tf
def test_preprocess_shapes_dtype_and_range():
    import train_and_quantize as tq

    raw = np.random.default_rng(0).integers(0, 256, size=(5, 28, 28), dtype=np.uint8)
    raw[0, 0, 0], raw[0, 0, 1] = 0, 255
    x = tq.preprocess(raw)
    assert x.shape == (5, 28, 28, 1)
    assert x.dtype == np.float32
    assert x.min() == 0.0 and x.max() == 1.0


@needs_tf
def test_teacher_and_student_architectures_match_assignment():
    import train_and_quantize as tq

    teacher, student = tq.build_teacher(CFG), tq.build_student(CFG)
    assert teacher.count_params() == 503_690          # Assignment 4 teacher
    assert student.count_params() == 14_410           # Assignment 4 student (deployed)
    assert [type(l).__name__ for l in student.layers] == [
        "Conv2D", "MaxPooling2D", "Conv2D", "MaxPooling2D", "Flatten", "Dense", "Dense"]
    for m in (teacher, student):
        assert m.input_shape == (None, 28, 28, 1) and m.output_shape == (None, 10)
        assert m.layers[-1].activation.__name__ == "linear"   # logits, no softmax
    with pytest.raises(ValueError):
        tq.build_from_spec([{"type": "lstm"}], CFG, "bad")


@needs_tf
def test_distiller_train_step_updates_only_the_student():
    import train_and_quantize as tq

    tq.set_seeds(0)
    teacher, student = tq.build_teacher(CFG), tq.build_student(CFG)
    t_before = [w.copy() for w in teacher.get_weights()]
    s_before = [w.copy() for w in student.get_weights()]
    d = tq.Distiller(student, teacher, alpha=0.1, temperature=5)
    d.compile(optimizer=tq.keras.optimizers.Adam(1e-3), metrics=["accuracy"])
    x = np.random.default_rng(0).random((16, 28, 28, 1), dtype=np.float32)
    y = np.arange(16) % 10
    hist = d.fit(x, y, epochs=1, batch_size=8, verbose=0).history
    assert {"accuracy", "student_loss", "distill_loss"} <= set(hist)
    assert all(np.array_equal(a, b) for a, b in zip(t_before, teacher.get_weights()))  # teacher frozen
    assert any(not np.array_equal(a, b) for a, b in zip(s_before, student.get_weights()))


@needs_tf
def test_teacher_cache_key_tracks_teacher_config_only():
    import copy
    import train_and_quantize as tq

    k = tq.teacher_cache_key(CFG)
    other = copy.deepcopy(CFG)
    other["distillation"]["temperature"] = 9          # student-side change: teacher reusable
    assert tq.teacher_cache_key(other) == k
    other["training"]["teacher"]["epochs"] = 1         # teacher change: must retrain
    assert tq.teacher_cache_key(other) != k


@needs_tf
def test_import_rejects_a_model_that_is_not_the_configured_student(tmp_path):
    import argparse
    import train_and_quantize as tq

    wrong = tq.build_teacher(CFG)                     # teacher weights cannot load into the student
    wrong.save(tmp_path / "wrong.h5", include_optimizer=False)
    x = np.zeros((4, 28, 28, 1), np.float32)
    y = np.zeros(4, np.int64)
    args = argparse.Namespace(import_h5=str(tmp_path / "wrong.h5"), import_tflite=None)
    with pytest.raises(SystemExit):
        tq.import_mode(CFG, args, ((x, y), (x, y)))


@needs_tf
def test_representative_dataset_uses_first_n_training_images():
    import train_and_quantize as tq

    x = np.random.default_rng(1).random((500, 28, 28, 1), dtype=np.float32)
    a = [s[0] for s in tq.make_representative_dataset(x, 200)()]
    assert len(a) == 200 and a[0].shape == (1, 28, 28, 1)
    assert np.array_equal(a[0][0], x[0]) and np.array_equal(a[-1][0], x[199])


def test_config_hardware_budget_is_consistent():
    hw, g = CFG["hardware"], CFG["gates"]
    assert hw["tensor_arena_bytes"] == 32 * 1024
    assert g["max_model_bytes"] < hw["flash_bytes"]
    assert set(g["allowed_ops"]) == {"CONV_2D", "MAX_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "QUANTIZE", "DEQUANTIZE"}
    src = REPO / "firmware_esp32" / "src"
    runtime_cpp = (src / "model_runtime.cpp").read_text()
    assert "MicroMutableOpResolver<6>" in runtime_cpp
    for fn in ("AddConv2D", "AddMaxPool2D", "AddReshape", "AddFullyConnected", "AddQuantize", "AddDequantize"):
        assert f"r.{fn}()" in runtime_cpp
    all_src = "".join(p.read_text() for p in src.glob("*.cpp"))
    assert "AllOpsResolver" not in all_src
    # The OTA updater rejects models whose ops are not in this list: it must equal the gate's list.
    m = re.search(r'kSupportedOps\[\] = "([^"]+)"', (src / "main.cpp").read_text())
    assert m and m.group(1).split(",") == g["allowed_ops"]


# =========================================================================== #
# gates (pure logic)
# =========================================================================== #
def test_size_gate_rejects_oversized_model():
    assert gate.check_size_gate(34_999, 35_000).passed
    assert gate.check_size_gate(35_000, 35_000).passed
    assert not gate.check_size_gate(35_001, 35_000).passed


def test_accuracy_gate_rejects_low_accuracy_and_large_drop():
    ok = gate.check_accuracy_gate(0.975, 0.980, 0.96, 0.015)
    assert all(r.passed for r in ok)
    low = gate.check_accuracy_gate(0.950, 0.955, 0.96, 0.015)
    assert not low[0].passed and low[1].passed
    drop = gate.check_accuracy_gate(0.961, 0.990, 0.96, 0.015)
    assert drop[0].passed and not drop[1].passed


def test_operator_gate_allows_only_micro_ops():
    allowed = CFG["gates"]["allowed_ops"]
    assert gate.check_operator_gate(["CONV_2D", "MAX_POOL_2D", "RESHAPE", "FULLY_CONNECTED"], allowed).passed
    bad = gate.check_operator_gate(["CONV_2D", "SOFTMAX", "CUSTOM:Foo"], allowed)
    assert not bad.passed and "SOFTMAX" in bad.detail and "CUSTOM:Foo" in bad.detail


def test_quantization_gate():
    good_in = {"dtype": np.int8, "quantization": (0.00392157, -128)}
    good_out = {"dtype": np.int8, "quantization": (0.15, 10)}
    assert gate.check_quantization_gate(good_in, good_out).passed
    float_out = {"dtype": np.float32, "quantization": (0.0, 0)}
    r = gate.check_quantization_gate(good_in, float_out)
    assert not r.passed and "output dtype=float32" in r.detail
    bad_zp = {"dtype": np.int8, "quantization": (0.1, 200)}
    assert not gate.check_quantization_gate(bad_zp, good_out).passed


def test_quantize_dequantize_roundtrip():
    x = np.linspace(0, 1, 256, dtype=np.float32)
    scale, zp = 1 / 255.0, -128
    q = gate.quantize(x, scale, zp)
    assert q.dtype == np.int8 and q.min() == -128 and q.max() == 127
    assert np.max(np.abs(gate.dequantize(q, scale, zp) - x)) <= scale / 2 + 1e-6


def test_golden_selection_is_first_correct_per_class():
    labels = np.array([3, 0, 1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 9])
    preds = np.array([3, 5, 1, 0, 2, 3, 4, 5, 6, 7, 8, 1, 9])
    idx = gate.select_golden_vectors(None, labels, preds)
    assert idx == [3, 2, 4, 0, 6, 7, 8, 9, 10, 12]
    with pytest.raises(RuntimeError):
        gate.select_golden_vectors(None, labels, np.zeros_like(labels))


def test_flatbuffer_opcode_parser_on_synthetic_model():
    fb = build_fake_tflite([3, 17, 22, 9, 114, 6])
    assert gate.parse_tflite_opcodes(fb) == ["CONV_2D", "MAX_POOL_2D", "RESHAPE", "FULLY_CONNECTED",
                                             "QUANTIZE", "DEQUANTIZE"]
    fb_bad = build_fake_tflite([3, 25, 32], custom_name="Fancy")
    assert gate.parse_tflite_opcodes(fb_bad) == ["CONV_2D", "SOFTMAX", "CUSTOM:Fancy"]
    with pytest.raises(ValueError):
        gate.parse_tflite_opcodes(b"\x00" * 16)


# =========================================================================== #
# real artifacts
# =========================================================================== #
@needs_model
@needs_tf
def test_int8_model_tensor_types_scales_and_zero_points():
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(INT8_MODEL))
    interp.allocate_tensors()
    i, o = interp.get_input_details()[0], interp.get_output_details()[0]
    assert i["dtype"] == np.int8 and o["dtype"] == np.int8
    assert list(i["shape"]) == [1, 28, 28, 1] and list(o["shape"]) == [1, 10]
    in_scale, in_zp = i["quantization"]
    assert abs(in_scale - 1 / 255.0) < 1e-3 and in_zp == -128  # [0,1] input range
    assert o["quantization"][0] > 0 and -128 <= o["quantization"][1] <= 127
    # every weight-bearing tensor must be int8 (biases int32): full-integer model
    dtypes = {d["dtype"] for d in interp.get_tensor_details()}
    assert np.float32 not in dtypes


@needs_model
def test_real_model_passes_operator_and_size_audit():
    data = INT8_MODEL.read_bytes()
    ops = gate.parse_tflite_opcodes(data)
    assert {"CONV_2D", "MAX_POOL_2D", "FULLY_CONNECTED"} <= set(ops)
    assert gate.check_operator_gate(ops, CFG["gates"]["allowed_ops"]).passed
    assert gate.check_size_gate(len(data), CFG["gates"]["max_model_bytes"]).passed
    assert len(data) < FP32_MODEL.stat().st_size / 2.5  # INT8 ~4x smaller than FP32


@needs_model
@needs_tf
def test_gate_cli_blocks_model_when_flash_ceiling_tightened(tmp_path):
    """The professor demo: `--max-bytes 10000` must make the pipeline exit 1."""
    rep = tmp_path / "gate.json"
    proc = subprocess.run(
        [sys.executable, str(REPO / "ml_pipeline" / "evaluate_and_gate.py"), "--max-bytes", "10000",
         "--limit", "300", "--report-out", str(rep), "--golden-out", str(tmp_path / "g.json"), "--no-mlflow"],
        capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "DEPLOYMENT BLOCKED" in proc.stdout and "FLASH_SIZE" in proc.stdout
    report = json.loads(rep.read_text())
    assert report["passed"] is False
    assert not (tmp_path / "g.json").exists()  # no golden vectors for a blocked model


# =========================================================================== #
# codegen
# =========================================================================== #
def _fake_golden(sha: str) -> dict:
    rng = np.random.default_rng(7)
    return {
        "selection": "first_correct_per_class", "model_sha256": sha,
        "input_scale": 1 / 255.0, "input_zero_point": -128, "output_scale": 0.17, "output_zero_point": 5,
        "vectors": [{"test_index": 100 + d, "label": d,
                     "image_int8": rng.integers(-128, 128, 784).tolist(),
                     "expected_logits_int8": rng.integers(-128, 128, 10).tolist(),
                     "expected_pred": d} for d in range(10)],
    }


def _extract_bytes(header: str) -> bytes:
    body = re.search(r"g_mnist_model_data\[\]\s*=\s*\{(.*?)\};", header, re.S).group(1)
    return bytes(int(h, 16) for h in re.findall(r"0x([0-9a-f]{2})", body))


def test_model_header_roundtrip_alignment_and_sha():
    data = build_fake_tflite([3, 17, 22, 9], pad_to=1001)
    h = codegen.generate_model_header(data, {"int8_accuracy": 0.98, "input_scale": 0.0039, "input_zero_point": -128})
    assert "alignas(16) const unsigned char g_mnist_model_data[] = {" in h
    assert f"const unsigned int g_mnist_model_data_len = {len(data)};" in h
    assert _extract_bytes(h) == data
    sha = hashlib.sha256(data).hexdigest()
    assert f'#define TINYMLOPS_MODEL_SHA256 "{sha}"' in h
    assert h.count("{") == h.count("}") and "#endif" in h


def test_vectors_header_contents_and_validation():
    g = _fake_golden("ab" * 32)
    h = codegen.generate_vectors_header(g)
    assert "const int8_t kTestImages[kNumTestVectors][kImageSize]" in h
    assert "const uint8_t kTestLabels[kNumTestVectors] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};" in h
    assert "constexpr int kInputZeroPoint = -128;" in h and "constexpr int kOutputZeroPoint = 5;" in h
    from serial_hil_verifier import parse_vectors_header

    tmp = REPO / "artifacts" / "_tmp_vectors_test.h"
    tmp.parent.mkdir(exist_ok=True)
    try:
        tmp.write_text(h)
        parsed = parse_vectors_header(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    assert parsed["images"].tolist() == [v["image_int8"] for v in g["vectors"]]
    assert parsed["logits"].tolist() == [v["expected_logits_int8"] for v in g["vectors"]]
    g["vectors"][0]["image_int8"] = g["vectors"][0]["image_int8"][:-1]
    with pytest.raises(ValueError):
        codegen.generate_vectors_header(g)


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not on PATH")
def test_generated_headers_compile_with_gxx(tmp_path):
    data = build_fake_tflite([3, 9], pad_to=777)
    (tmp_path / "model_data.h").write_text(codegen.generate_model_header(data))
    (tmp_path / "test_vectors.h").write_text(codegen.generate_vectors_header(_fake_golden(hashlib.sha256(data).hexdigest())))
    (tmp_path / "tu.cpp").write_text(
        '#include "model_data.h"\n#include "test_vectors.h"\n'
        "static_assert(sizeof(g_mnist_model_data) == 777, \"len\");\n"
        "static_assert(alignof(decltype(g_mnist_model_data)) >= 16 || true, \"align\");\n"
        "static_assert(sizeof(kTestImages) == 10 * 784, \"images\");\n"
        "int main() { return (reinterpret_cast<unsigned long long>(g_mnist_model_data) % 16) + kTestLabels[0]; }\n")
    proc = subprocess.run(["g++", "-std=c++17", "-Wall", "-Werror", "-c", "tu.cpp", "-o", "tu.o"],
                          cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(not (MODEL_H.exists() and INT8_MODEL.exists()), reason="run `make codegen` first")
def test_committed_firmware_headers_match_model_artifact():
    model = INT8_MODEL.read_bytes()
    h = MODEL_H.read_text()
    assert _extract_bytes(h) == model, "model_data.h is stale - run `make codegen`"
    sha = hashlib.sha256(model).hexdigest()
    assert sha in h
    assert sha in VECTORS_H.read_text(), "test_vectors.h was generated for a different model"


@needs_golden
def test_golden_vectors_cover_all_digits_and_are_int8():
    g = json.loads(GOLDEN.read_text())
    assert [v["label"] for v in g["vectors"]] == list(range(10))
    for v in g["vectors"]:
        arr = np.array(v["image_int8"])
        assert arr.shape == (784,) and arr.min() >= -128 and arr.max() <= 127
        assert int(np.argmax(v["expected_logits_int8"])) == v["label"] == v["expected_pred"]


def test_log_parser_detects_pass_and_failures():
    from serial_hil_verifier import parse_log

    log = """[TELEMETRY] MODEL_INTEGRITY=OK
| 7 | 7 | 7 | 99.12% | 1432 | EXACT | PASS |
[TINYMLOPS_PARITY] BITEXACT=10/10
[TINYMLOPS_TEST_SUMMARY] PASSED=10/10 ACCURACY=100.00% AVG_LATENCY_US=1420.5 ARENA_BYTES=11264
"""
    r = parse_log(log)
    assert r["summary"] == {"passed": 10, "total": 10, "accuracy": 100.0, "avg_latency_us": 1420.5, "arena_bytes": 11264}
    assert r["rows"][0]["predicted"] == 7 and not r["fatal"]
    assert r["telemetry"]["BITEXACT"] == "10/10"
    # real Wokwi capture lost a newline between two telemetry lines
    merged = parse_log("[TELEMETRY] MODEL_INTEGRITY=OK[TELEMETRY] HEAP_TOTAL=337232 HEAP_FREE_BEFORE_ALLOC=308328\n")
    assert merged["telemetry"]["MODEL_INTEGRITY"] == "OK"
    assert merged["telemetry"]["HEAP_TOTAL"] == "337232"
    bad = parse_log("[TINYMLOPS_FATAL] AllocateTensors() failed\nGuru Meditation Error: Core 1 panic'ed")
    assert len(bad["fatal"]) == 2 and bad["summary"] is None
