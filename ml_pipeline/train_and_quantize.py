#!/usr/bin/env python3
"""
Stage 1 of the TinyMLOps pipeline: train + quantize.

  1. Load MNIST (60k train / 10k test), normalise to [0, 1] float32, shape (28, 28, 1).
  2. Train the FP32 baseline CNN and export artifacts/baseline_fp32.tflite.
  3. Wrap the baseline with tfmot quantize_model (Quantization-Aware Training),
     fine-tune, and convert to a *full-integer* INT8 flatbuffer
     (int8 input, int8 output, TFLITE_BUILTINS_INT8 only) ->
     artifacts/model_qat_int8.tflite.
  4. Log params / sizes / accuracies to MLflow (./mlruns) and write
     artifacts/metrics.json (tracked by DVC as a metrics file).

Usage:
    python ml_pipeline/train_and_quantize.py [--config ml_pipeline/config.yaml]
                                             [--epochs N] [--qat-epochs N]
"""
from __future__ import annotations

import os

# Must be set BEFORE tensorflow / tfmot are imported so that tf.keras resolves to
# tf_keras (Keras 2). tfmot's quantize_model does not support Keras 3 models.
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml
import tensorflow as tf
import tensorflow_model_optimization as tfmot

keras = tf.keras  # == tf_keras because of TF_USE_LEGACY_KERAS=1

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "ml_pipeline" / "config.yaml"


# --------------------------------------------------------------------------- #
# Config / data helpers (imported by the tests)
# --------------------------------------------------------------------------- #
def load_config(path: str | Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def preprocess(images: np.ndarray) -> np.ndarray:
    """uint8 (N, 28, 28) -> float32 (N, 28, 28, 1) in [0, 1]."""
    x = images.astype(np.float32) / 255.0
    if x.ndim == 3:
        x = x[..., np.newaxis]
    return x


def load_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Returns ((x_train, y_train), (x_test, y_test)) already preprocessed.

    Keras caches the archive at ~/.keras/datasets/mnist.npz (cached in CI).
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return (preprocess(x_train), y_train.astype(np.int64)), (preprocess(x_test), y_test.astype(np.int64))


class _IntSafeRandom(random.Random):
    """tf_keras 2.16 calls randint(1, 1e9) with a float bound, which Python >= 3.12 rejects."""

    def randint(self, a, b):  # noqa: D401
        return super().randint(int(a), int(b))


def set_seeds(seed: int) -> None:
    """Equivalent of keras.utils.set_random_seed(seed) that also works on Python 3.12+."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        from tf_keras.src import backend as _kb

        _kb._SEED_GENERATOR.generator = _IntSafeRandom(seed)
    except ImportError:  # other Keras 2 packaging: fall back to the public API
        keras.utils.set_random_seed(seed)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(cfg: Dict[str, Any]) -> keras.Model:
    m = cfg["model"]
    return keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=tuple(cfg["data"]["input_shape"])),
            keras.layers.Conv2D(m["conv_filters"], tuple(m["conv_kernel"]), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=tuple(m["pool_size"])),
            keras.layers.Flatten(),
            keras.layers.Dense(cfg["data"]["num_classes"]),  # logits
        ],
        name="mnist_tiny_cnn",
    )


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def convert_fp32(model: keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def make_representative_dataset(x_train: np.ndarray, n: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_train), size=n, replace=False)

    def representative_dataset():
        for i in idx:
            yield [x_train[i : i + 1].astype(np.float32)]

    return representative_dataset


def convert_int8(qat_model: keras.Model, representative_dataset) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# MLflow (optional at runtime: pipeline still works if mlflow is missing)
# --------------------------------------------------------------------------- #
def log_to_mlflow(cfg: Dict[str, Any], params: Dict[str, Any], metrics: Dict[str, float],
                  artifacts: list[Path]) -> str | None:
    try:
        import mlflow
    except ImportError:
        print("[mlflow] not installed - skipping experiment tracking")
        return None
    tracking_dir = resolve(cfg["paths"]["mlflow_tracking_uri"])
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(cfg["project"]["name"])
    with mlflow.start_run(run_name=f"train-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        mlflow.set_tags({"course": cfg["project"]["course"], "target": cfg["hardware"]["target"],
                         "stage": "train_and_quantize"})
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for a in artifacts:
            if a.exists():
                mlflow.log_artifact(str(a), artifact_path="models")
        return run.info.run_id


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--epochs", type=int, default=None, help="override baseline epochs")
    ap.add_argument("--qat-epochs", type=int, default=None, help="override QAT epochs")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed = int(cfg["project"]["seed"])
    set_seeds(seed)

    tb, tq = cfg["training"]["baseline"], cfg["training"]["qat"]
    base_epochs = args.epochs if args.epochs is not None else int(tb["epochs"])
    qat_epochs = args.qat_epochs if args.qat_epochs is not None else int(tq["epochs"])

    art_dir = resolve(cfg["paths"]["artifacts_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = resolve(cfg["paths"]["baseline_tflite"])
    int8_path = resolve(cfg["paths"]["int8_tflite"])
    metrics_path = resolve(cfg["paths"]["metrics_json"])

    print("=" * 72)
    print(" TinyMLOps | Stage 1: train FP32 baseline + QAT INT8")
    print(f" TensorFlow {tf.__version__} | tfmot {tfmot.__version__} | Python {platform.python_version()}")
    print("=" * 72)

    (x_train, y_train), (x_test, y_test) = load_mnist()
    print(f"[data] train={x_train.shape} test={x_test.shape} "
          f"range=[{x_train.min():.1f}, {x_train.max():.1f}] dtype={x_train.dtype}")

    # ---- FP32 baseline --------------------------------------------------- #
    baseline = build_model(cfg)
    compile_model(baseline, float(tb["learning_rate"]))
    baseline.summary(print_fn=lambda s: print("   " + s))
    t0 = time.time()
    hist_b = baseline.fit(x_train, y_train, batch_size=int(tb["batch_size"]), epochs=base_epochs,
                          validation_split=float(cfg["data"]["validation_split"]), verbose=2)
    base_train_s = time.time() - t0
    _, base_keras_acc = baseline.evaluate(x_test, y_test, verbose=0)
    print(f"[fp32] keras test accuracy = {base_keras_acc:.4f} ({base_train_s:.1f}s)")

    fp32_bytes = convert_fp32(baseline)
    fp32_path.write_bytes(fp32_bytes)
    print(f"[fp32] wrote {fp32_path.relative_to(REPO_ROOT)} ({len(fp32_bytes):,} bytes)")

    # ---- Quantization-Aware Training ------------------------------------- #
    qat_model = tfmot.quantization.keras.quantize_model(baseline)
    compile_model(qat_model, float(tq["learning_rate"]))
    t0 = time.time()
    hist_q = qat_model.fit(x_train, y_train, batch_size=int(tq["batch_size"]), epochs=qat_epochs,
                           validation_split=float(cfg["data"]["validation_split"]), verbose=2)
    qat_train_s = time.time() - t0
    _, qat_keras_acc = qat_model.evaluate(x_test, y_test, verbose=0)
    print(f"[qat ] keras (fake-quant) test accuracy = {qat_keras_acc:.4f} ({qat_train_s:.1f}s)")

    rep = make_representative_dataset(x_train, int(cfg["quantization"]["representative_samples"]), seed)
    int8_bytes = convert_int8(qat_model, rep)
    int8_path.write_bytes(int8_bytes)
    print(f"[int8] wrote {int8_path.relative_to(REPO_ROOT)} ({len(int8_bytes):,} bytes, "
          f"{len(fp32_bytes) / len(int8_bytes):.2f}x smaller than FP32)")

    params = {
        "seed": seed,
        "conv_filters": cfg["model"]["conv_filters"],
        "conv_kernel": "x".join(map(str, cfg["model"]["conv_kernel"])),
        "pool_size": "x".join(map(str, cfg["model"]["pool_size"])),
        "baseline_batch_size": tb["batch_size"],
        "baseline_epochs": base_epochs,
        "baseline_lr": tb["learning_rate"],
        "qat_batch_size": tq["batch_size"],
        "qat_epochs": qat_epochs,
        "qat_lr": tq["learning_rate"],
        "representative_samples": cfg["quantization"]["representative_samples"],
        "total_params": int(baseline.count_params()),
        "tf_version": tf.__version__,
    }
    metrics = {
        "fp32_keras_test_accuracy": float(base_keras_acc),
        "qat_keras_test_accuracy": float(qat_keras_acc),
        "fp32_final_val_accuracy": float(hist_b.history["val_accuracy"][-1]),
        "qat_final_val_accuracy": float(hist_q.history["val_accuracy"][-1]),
        "fp32_tflite_bytes": len(fp32_bytes),
        "int8_tflite_bytes": len(int8_bytes),
        "compression_ratio": round(len(fp32_bytes) / len(int8_bytes), 3),
        "baseline_train_seconds": round(base_train_s, 2),
        "qat_train_seconds": round(qat_train_s, 2),
    }
    run_id = log_to_mlflow(cfg, params, metrics, [fp32_path, int8_path])

    record = {
        "params": params,
        "metrics": metrics,
        "fp32_sha256": sha256(fp32_bytes),
        "int8_sha256": sha256(int8_bytes),
        "mlflow_run_id": run_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metrics_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[done] metrics -> {metrics_path.relative_to(REPO_ROOT)}  mlflow_run_id={run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
