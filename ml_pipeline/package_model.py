#!/usr/bin/env python3
"""
Stage 4 (release): package a gate-approved model for over-the-air delivery to the ESP32.

    python ml_pipeline/package_model.py --version 57 \
        --base-url https://github.com/Bathri007/tiny-mlops-project/releases/download/model-v57

writes dist/:
    model.tflite      the INT8 KD student, byte-identical to the one CI gated and simulated
    vectors.bin       its 10 golden INT8 test digits + host reference logits ("TMV1" pack, see
                      firmware_esp32/lib/tinymlops_core/vector_pack.h) for the on-device self-test
    manifest.json     version, URLs, sizes, SHA-256s, accuracy, min firmware, ops
    RELEASE_NOTES.md  human-readable summary for the GitHub Release

The device downloads manifest.json from .../releases/latest/download/manifest.json, then both files,
checks their SHA-256, self-tests the model and only then switches to it.

--demo-fault (for demonstrating the safety nets; never use for a real release):
    sha       model.tflite has one byte flipped AFTER hashing -> device rejects it (SHA_MISMATCH)
    selftest  vectors.bin labels rotated by one -> model fails its on-device self-test -> rollback
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "ml_pipeline" / "config.yaml"
MAGIC = b"TMV1"
HEADER = struct.Struct("<4sHHHHfifi")  # magic, count, image_size, num_classes, reserved, in_s, in_zp, out_s, out_zp
MIN_FIRMWARE = "2.0.0"


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_vector_pack(golden: Dict[str, Any]) -> bytes:
    """golden_vectors.json -> TMV1 bytes (layout documented in vector_pack.h)."""
    vecs = golden["vectors"]
    images = np.array([v["image_int8"] for v in vecs], dtype=np.int8)
    logits = np.array([v["expected_logits_int8"] for v in vecs], dtype=np.int8)
    labels = np.array([v["label"] for v in vecs], dtype=np.uint8)
    count, image_size = images.shape
    num_classes = logits.shape[1]
    header = HEADER.pack(MAGIC, count, image_size, num_classes, 0, float(golden["input_scale"]),
                         int(golden["input_zero_point"]), float(golden["output_scale"]),
                         int(golden["output_zero_point"]))
    return header + labels.tobytes() + logits.tobytes() + images.tobytes()


def read_vector_pack(data: bytes) -> Dict[str, Any]:
    magic, count, image_size, num_classes, _, in_s, in_zp, out_s, out_zp = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("bad magic")
    expect = HEADER.size + count * (1 + num_classes + image_size)
    if len(data) != expect:
        raise ValueError(f"size {len(data)} != {expect}")
    p = HEADER.size
    labels = np.frombuffer(data, np.uint8, count, p); p += count
    logits = np.frombuffer(data, np.int8, count * num_classes, p).reshape(count, num_classes); p += count * num_classes
    images = np.frombuffer(data, np.int8, count * image_size, p).reshape(count, image_size)
    return {"input_scale": in_s, "input_zero_point": in_zp, "output_scale": out_s, "output_zero_point": out_zp,
            "labels": labels, "logits": logits, "images": images}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--version", type=int, required=True, help="monotonic model version (CI: the run number)")
    ap.add_argument("--base-url", required=True, help="https URL the release assets will be downloadable from")
    ap.add_argument("--out", default=str(REPO_ROOT / "dist"))
    ap.add_argument("--git-sha", default="")
    ap.add_argument("--demo-fault", choices=["none", "sha", "selftest"], default="none")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    p = cfg["paths"]
    model_path, golden_path, gate_path = resolve(p["int8_tflite"]), resolve(p["golden_vectors_json"]), resolve(p["gate_report_json"])
    for f in (model_path, golden_path, gate_path):
        if not f.exists():
            print(f"[error] missing {f} - run train, gate and codegen first")
            return 1
    if not args.base_url.startswith("https://"):
        print("[error] --base-url must be https://")
        return 1

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        print("[error] gate_report.json says the model FAILED its quality gates - refusing to package")
        return 1
    model = model_path.read_bytes()
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    if golden.get("model_sha256") != sha256(model):
        print("[error] golden vectors were generated for a different model - re-run the gate")
        return 1

    vectors = write_vector_pack(golden)
    model_sha, vectors_sha = sha256(model), sha256(vectors)
    if args.demo_fault == "sha":
        model = model[:100] + bytes([model[100] ^ 0xFF]) + model[101:]          # hash no longer matches
    elif args.demo_fault == "selftest":
        pack = read_vector_pack(vectors)
        golden_bad = dict(golden, vectors=[dict(v, label=(v["label"] + 1) % 10) for v in golden["vectors"]])
        vectors = write_vector_pack(golden_bad)                                  # wrong expected labels
        vectors_sha = sha256(vectors)
        assert len(pack["labels"]) == len(golden_bad["vectors"])

    base = args.base_url.rstrip("/")
    manifest = {
        "schema": 1,
        "version": args.version,
        "tag": f"model-v{args.version}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": args.git_sha,
        "model_source": gate.get("model_source", "unknown"),
        "model_url": f"{base}/model.tflite",
        "model_sha256": model_sha,
        "model_size": len(model),
        "vectors_url": f"{base}/vectors.bin",
        "vectors_sha256": vectors_sha,
        "vectors_size": len(vectors),
        "int8_accuracy": round(float(gate["metrics"]["int8_tflite_accuracy"]), 6),
        "min_firmware": MIN_FIRMWARE,
        "ops": ",".join(gate["operators"]),
        "demo_fault": args.demo_fault,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.tflite").write_bytes(model)
    (out / "vectors.bin").write_bytes(vectors)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    kd = gate.get("kd", {})
    notes = [f"## KD student model v{args.version}", "",
             f"- INT8 accuracy (10,000 MNIST test images): **{100 * manifest['int8_accuracy']:.2f}%**",
             f"- Size: {len(model):,} bytes · operators: {manifest['ops']}",
             f"- Model SHA-256: `{model_sha}`",
             f"- Source: `{manifest['model_source']}` · commit `{args.git_sha[:7] or 'n/a'}`"]
    if "kd_student_keras_test_accuracy" in kd:
        notes.append(f"- KD student FP32 {100 * kd['kd_student_keras_test_accuracy']:.2f}%"
                     + (f" (+{100 * kd['kd_gain_vs_baseline_student']:.2f} pp vs baseline student)"
                        if "kd_gain_vs_baseline_student" in kd else ""))
    notes += ["", "Passed every quality gate and the Wokwi ESP32 simulation before publishing. Devices running "
              "firmware >= 2.0.0 download it over WiFi, verify its SHA-256 and self-test it before switching."]
    if args.demo_fault != "none":
        notes += ["", f"**DEMO FAULT `{args.demo_fault}`: this release is deliberately broken; devices must reject it.**"]
    (out / "RELEASE_NOTES.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"[package] v{args.version} -> {out}  model={len(model):,} B sha={model_sha[:16]}...  "
          f"vectors={len(vectors):,} B  demo_fault={args.demo_fault}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
