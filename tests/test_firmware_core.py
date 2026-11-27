"""
Host tests for the ESP32 over-the-air (OTA) model update path.

The firmware's platform-independent core (firmware_esp32/lib/tinymlops_core: SHA-256, the TMV1
vector pack, the manifest reader and the update decision) is compiled with g++ into a small CLI
and checked against the Python side that produces releases (ml_pipeline/package_model.py).
So the exact C++ the ESP32 runs is tested without hardware, locally and in CI.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / "firmware_esp32" / "lib" / "tinymlops_core"
sys.path.insert(0, str(REPO / "ml_pipeline"))

import package_model as pkg  # noqa: E402

GXX = shutil.which("g++")
pytestmark = pytest.mark.skipif(GXX is None, reason="g++ not on PATH")
FW = "2.0.0"
OPS = "CONV_2D,MAX_POOL_2D,RESHAPE,FULLY_CONNECTED,QUANTIZE,DEQUANTIZE"


@pytest.fixture(scope="module")
def cli(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("core") / ("core_cli.exe" if os.name == "nt" else "core_cli")
    srcs = [str(REPO / "tests" / "host_cpp" / "core_cli.cpp")] + [str(p) for p in sorted(CORE.glob("*.cpp"))]
    proc = subprocess.run([GXX, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-I", str(CORE), *srcs, "-o",
                           str(out)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return out


def run(cli: Path, *args) -> str:
    proc = subprocess.run([str(cli), *map(str, args)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def fake_golden(count=10, image_size=784, classes=10, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    return {"input_scale": 1 / 255.0, "input_zero_point": -128, "output_scale": 0.19169481, "output_zero_point": -16,
            "vectors": [{"label": i % 10, "image_int8": rng.integers(-128, 128, image_size).tolist(),
                         "expected_logits_int8": rng.integers(-128, 128, classes).tolist()} for i in range(count)]}


# --------------------------------------------------------------------------- SHA-256
@pytest.mark.parametrize("size", [0, 1, 55, 56, 63, 64, 65, 119, 120, 1000, 18816])
@pytest.mark.parametrize("chunk", [1, 7, 64, 1024])
def test_streaming_sha256_matches_hashlib(cli, tmp_path, size, chunk):
    data = np.random.default_rng(size).integers(0, 256, size, dtype=np.uint8).tobytes()
    f = tmp_path / "blob.bin"
    f.write_bytes(data)
    assert run(cli, "sha", f, chunk) == hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- TMV1 vector pack
def test_vector_pack_python_to_cpp_roundtrip(cli, tmp_path):
    golden = fake_golden()
    data = pkg.write_vector_pack(golden)
    assert len(data) == 28 + 10 * (1 + 10 + 784)
    f = tmp_path / "vectors.bin"
    f.write_bytes(data)
    got = json.loads(run(cli, "pack", f))
    imgs = np.array([v["image_int8"] for v in golden["vectors"]])
    logits = np.array([v["expected_logits_int8"] for v in golden["vectors"]])
    assert got["count"] == 10 and got["image_size"] == 784 and got["num_classes"] == 10
    assert got["labels"] == [v["label"] for v in golden["vectors"]]
    assert got["image_sum"] == int(imgs.sum()) and got["logit_sum"] == int(logits.sum())
    assert got["last_pixel"] == int(imgs[-1, -1])
    assert got["input_zero_point"] == -128 and got["output_zero_point"] == -16
    assert abs(got["input_scale"] - 1 / 255.0) < 1e-9 and abs(got["output_scale"] - 0.19169481) < 1e-7
    back = pkg.read_vector_pack(data)  # python reader agrees too
    assert np.array_equal(back["images"], imgs) and np.array_equal(back["logits"], logits)


@pytest.mark.parametrize("mutate,err", [
    (lambda d: d[:20], "too_short"),
    (lambda d: b"XXXX" + d[4:], "bad_magic"),
    (lambda d: d[:-1], "size_mismatch"),
    (lambda d: d[:4] + (0).to_bytes(2, "little") + d[6:], "bad_counts"),
])
def test_vector_pack_rejects_corrupt_files(cli, tmp_path, mutate, err):
    f = tmp_path / "bad.bin"
    f.write_bytes(mutate(pkg.write_vector_pack(fake_golden())))
    assert json.loads(run(cli, "pack", f)) == {"error": err}


# --------------------------------------------------------------------------- manifest + decision
def manifest(**over) -> dict:
    m = {"schema": 1, "version": 57, "tag": "model-v57", "created_at": "2026-11-27T10:00:00Z", "git_sha": "abc",
         "model_source": "pipeline_train",
         "model_url": "https://github.com/o/r/releases/download/model-v57/model.tflite",
         "model_sha256": "a" * 64, "model_size": 18816,
         "vectors_url": "https://github.com/o/r/releases/download/model-v57/vectors.bin",
         "vectors_sha256": "b" * 64, "vectors_size": 7978, "int8_accuracy": 0.9825, "min_firmware": "2.0.0",
         "ops": "CONV_2D,MAX_POOL_2D,RESHAPE,FULLY_CONNECTED", "demo_fault": "none"}
    m.update(over)
    return m


def decide(cli, tmp_path, m, cur_ver=0, cur_sha="0" * 64, fw=FW, ops=OPS, max_bytes=200000, raw=None):
    f = tmp_path / "manifest.json"
    f.write_text(raw if raw is not None else json.dumps(m, indent=2))
    return json.loads(run(cli, "manifest", f, cur_ver, cur_sha, fw, ops, max_bytes))


def test_manifest_parsed_exactly(cli, tmp_path):
    r = decide(cli, tmp_path, manifest())
    assert r["parse"] == "ok" and r["version"] == 57 and r["tag"] == "model-v57" and r["model_size"] == 18816
    assert r["model_sha256"] == "a" * 64 and r["model_url"].endswith("/model-v57/model.tflite")
    assert abs(r["int8_accuracy"] - 0.9825) < 1e-6


@pytest.mark.parametrize("kw,action,reason", [
    (dict(), "update", "newer_version"),                                     # factory v0 -> v57
    (dict(cur_ver=57, cur_sha="a" * 64), "up_to_date", "same_model"),
    (dict(cur_ver=60), "up_to_date", "not_newer"),                           # never downgrade
    (dict(fw="1.9.9"), "reject", "firmware_too_old"),
    (dict(ops="CONV_2D,MAX_POOL_2D"), "reject", "unsupported_ops"),
    (dict(max_bytes=10000), "reject", "model_too_large"),
])
def test_update_decision(cli, tmp_path, kw, action, reason):
    r = decide(cli, tmp_path, manifest(), **kw)
    assert (r["action"], r["reason"]) == (action, reason)


@pytest.mark.parametrize("m,err", [
    (manifest(schema=2), "bad_schema"),
    (manifest(model_sha256="xyz"), "bad_sha"),
    (manifest(model_url="http://insecure.example/model.tflite"), "bad_url"),
    (manifest(version=0), "bad_size"),
    ({k: v for k, v in manifest().items() if k != "vectors_url"}, "missing_field"),
])
def test_manifest_rejects_bad_input(cli, tmp_path, m, err):
    assert decide(cli, tmp_path, m)["parse"] == err


def test_manifest_key_inside_a_string_value_is_not_matched(cli, tmp_path):
    m = manifest(git_sha='"version":999')  # a key-looking string must not shadow the real key
    assert decide(cli, tmp_path, m)["version"] == 57


@pytest.mark.parametrize("a,b,sign", [("2.0.0", "2.0.0", 0), ("2.0.1", "2.0.0", 1), ("2.0", "2.0.0", 0),
                                      ("1.10.0", "1.9.9", 1), ("2.0.0", "10.0.0", -1)])
def test_semver_compare(cli, a, b, sign):
    assert int(run(cli, "cmpver", a, b)) == sign


# --------------------------------------------------------------------------- package_model.py end to end
needs_artifacts = pytest.mark.skipif(
    not (REPO / "artifacts" / "kd_student_int8.tflite").exists() or not (REPO / "artifacts" / "gate_report.json").exists(),
    reason="run the pipeline first")


@needs_artifacts
@pytest.mark.parametrize("fault", ["none", "sha", "selftest"])
def test_package_model_release_is_understood_by_the_device_code(cli, tmp_path, fault):
    base = "https://github.com/o/r/releases/download/model-v7"
    assert pkg.main(["--version", "7", "--base-url", base, "--out", str(tmp_path), "--demo-fault", fault]) == 0
    man = json.loads((tmp_path / "manifest.json").read_text())
    model = (tmp_path / "model.tflite").read_bytes()
    vectors = (tmp_path / "vectors.bin").read_bytes()
    r = decide(cli, tmp_path, man)
    assert r["parse"] == "ok" and r["action"] == "update" and r["model_url"] == base + "/model.tflite"
    sha_ok = run(cli, "sha", tmp_path / "model.tflite", 1024) == man["model_sha256"]
    assert sha_ok == (fault != "sha")                         # 'sha' fault: device must see SHA_MISMATCH
    assert run(cli, "sha", tmp_path / "vectors.bin", 1024) == man["vectors_sha256"] == hashlib.sha256(vectors).hexdigest()
    pack = json.loads(run(cli, "pack", tmp_path / "vectors.bin"))
    golden = json.loads((REPO / "artifacts" / "golden_vectors.json").read_text())
    true_labels = [v["label"] for v in golden["vectors"]]
    assert (pack["labels"] == true_labels) == (fault != "selftest")  # 'selftest' fault: BIST must fail
    assert man["model_size"] == len(model) and man["vectors_size"] == len(vectors)
