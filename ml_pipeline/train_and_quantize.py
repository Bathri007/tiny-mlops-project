#!/usr/bin/env python3
"""
Stage 1 of the TinyMLOps pipeline: knowledge distillation + INT8 quantization.

TRAIN mode (default; what DVC and CI run):
  1. Load MNIST (60k train / 10k test), normalise to [0, 1] float32, shape (28, 28, 1).
  2. Teacher CNN (~504k params): trained once per teacher config and cached under
     artifacts/teacher_cache/ (keyed by a hash of its architecture + training config).
  3. Baseline student (~14k params) trained on hard labels only  -> KD control.
  4. KD student (same architecture) trained by the Distiller:
         loss = alpha * CE(y, student) + (1 - alpha) * T^2 * KL(softmax(t/T) || softmax(s/T))
  5. Export the KD student as FP32 .tflite (reference) and full-integer INT8 .tflite
     (post-training quantization, int8 in/out, TFLITE_BUILTINS_INT8) - the ESP32 model.
  6. Log params / metrics / models to MLflow (./mlruns) and write artifacts/metrics.json.

IMPORT mode (push the exact Colab Assignment 4 model through the same pipeline):
     python ml_pipeline/train_and_quantize.py --import-h5 models/colab/student_kd.h5 \
            [--import-tflite models/colab/student_kd_int8.tflite]
  The Keras .h5 provides the FP32 reference; the .tflite (if given) is shipped byte-for-byte,
  otherwise the .h5 is quantized here with the same PTQ settings.
"""
from __future__ import annotations

import os

# Must be set BEFORE tensorflow is imported so that tf.keras resolves to
# tf_keras (Keras 2), which the custom Distiller train_step and .h5 models rely on.
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
import tensorflow as tf

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


def _rel(p: Path) -> str:
    """Repo-relative path for log lines (absolute if outside the repo)."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def preprocess(images: np.ndarray) -> np.ndarray:
    """uint8 (N, 28, 28) -> float32 (N, 28, 28, 1) in [0, 1]."""
    x = images.astype(np.float32) / 255.0
    if x.ndim == 3:
        x = x[..., np.newaxis]
    return x


def load_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """((x_train, y_train), (x_test, y_test)), preprocessed. Cached at ~/.keras/datasets."""
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
# Models
# --------------------------------------------------------------------------- #
def build_from_spec(spec: List[Dict[str, Any]], cfg: Dict[str, Any], name: str) -> keras.Model:
    layers: List[keras.layers.Layer] = [keras.layers.InputLayer(input_shape=tuple(cfg["data"]["input_shape"]))]
    for item in spec:
        kind = item["type"]
        if kind == "conv":
            layers.append(keras.layers.Conv2D(int(item["filters"]), 3, activation="relu"))
        elif kind == "pool":
            layers.append(keras.layers.MaxPooling2D())
        elif kind == "flatten":
            layers.append(keras.layers.Flatten())
        elif kind == "dense":
            layers.append(keras.layers.Dense(int(item["units"]), activation="relu"))
        elif kind == "logits":
            layers.append(keras.layers.Dense(int(cfg["data"]["num_classes"])))  # no softmax
        else:
            raise ValueError(f"unknown layer type {kind!r} in models.{name}")
    return keras.Sequential(layers, name=name)


def build_teacher(cfg: Dict[str, Any]) -> keras.Model:
    return build_from_spec(cfg["models"]["teacher"], cfg, "teacher")


def build_student(cfg: Dict[str, Any], name: str = "student") -> keras.Model:
    return build_from_spec(cfg["models"]["student"], cfg, name)


def compile_supervised(model: keras.Model, lr: float) -> None:
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])


class Distiller(keras.Model):
    """Response-based knowledge distillation (Hinton et al.), as in the Assignment 4 notebook."""

    def __init__(self, student: keras.Model, teacher: keras.Model, alpha: float = 0.1, temperature: float = 5.0):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.alpha = alpha
        self.temperature = temperature
        self.student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.distill_loss_fn = keras.losses.KLDivergence()

    def call(self, x, training=False):
        return self.student(x, training=training)

    def train_step(self, data):
        x, y = data
        teacher_logits = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)
            student_loss = self.student_loss_fn(y, student_logits)
            t = self.temperature
            distill_loss = self.distill_loss_fn(tf.nn.softmax(teacher_logits / t, axis=1),
                                                tf.nn.softmax(student_logits / t, axis=1)) * (t ** 2)
            loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.compiled_metrics.update_state(y, student_logits)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss, "distill_loss": distill_loss})
        return results

    def test_step(self, data):
        x, y = data
        self.compiled_metrics.update_state(y, self.student(x, training=False))
        return {m.name: m.result() for m in self.metrics}


def teacher_cache_key(cfg: Dict[str, Any]) -> str:
    """Hash of everything that determines the trained teacher."""
    blob = json.dumps({"arch": cfg["models"]["teacher"], "train": cfg["training"]["teacher"],
                       "seed": cfg["project"]["seed"], "data": cfg["data"], "tf": tf.__version__},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def get_or_train_teacher(cfg, x_train, y_train, use_cache: bool = True) -> Tuple[keras.Model, bool, float]:
    """Returns (teacher, loaded_from_cache, train_seconds)."""
    cache = resolve(cfg["paths"]["teacher_cache_dir"]) / f"teacher_{teacher_cache_key(cfg)}.h5"
    if use_cache and cache.exists():
        print(f"[teacher] cache hit -> {_rel(cache)}")
        return keras.models.load_model(cache, compile=False), True, 0.0
    tt = cfg["training"]["teacher"]
    teacher = build_teacher(cfg)
    compile_supervised(teacher, float(tt["learning_rate"]))
    print(f"[teacher] training {teacher.count_params():,} params for {tt['epochs']} epochs (cache miss)")
    t0 = time.time()
    teacher.fit(x_train, y_train, epochs=int(tt["epochs"]), batch_size=int(tt["batch_size"]),
                validation_split=float(cfg["data"]["validation_split"]), verbose=2)
    seconds = time.time() - t0
    cache.parent.mkdir(parents=True, exist_ok=True)
    teacher.save(cache, include_optimizer=False)
    return teacher, False, seconds


def test_accuracy(model: keras.Model, x, y) -> float:
    logits = model.predict(x, batch_size=1000, verbose=0)
    return float((logits.argmax(1) == y).mean())


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def convert_fp32(model: keras.Model) -> bytes:
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def make_representative_dataset(x_train: np.ndarray, n: int):
    """First n training images, exactly like the Assignment 4 notebook."""
    def representative_dataset():
        for i in range(n):
            yield [x_train[i : i + 1].astype(np.float32)]

    return representative_dataset


def convert_int8_ptq(model: keras.Model, representative_dataset) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# MLflow (optional at runtime)
# --------------------------------------------------------------------------- #
def log_to_mlflow(cfg, params, metrics, artifacts: List[Path], run_name: str) -> Optional[str]:
    try:
        import mlflow
    except ImportError:
        print("[mlflow] not installed - skipping experiment tracking")
        return None
    tracking_dir = resolve(cfg["paths"]["mlflow_tracking_uri"])
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(cfg["project"]["name"])
    with mlflow.start_run(run_name=f"{run_name}-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        mlflow.set_tags({"course": cfg["project"]["course"], "target": cfg["hardware"]["target"],
                         "stage": "train_and_quantize", "model_source": params.get("model_source", "")})
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        for a in artifacts:
            if a.exists():
                mlflow.log_artifact(str(a), artifact_path="models")
        return run.info.run_id


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def train_mode(cfg, args, data) -> Tuple[keras.Model, Dict[str, Any], Dict[str, Any]]:
    (x_train, y_train), (x_test, y_test) = data
    vs = float(cfg["data"]["validation_split"])
    dcfg = cfg["distillation"]

    # ---- teacher (cached) ---------------------------------------------------
    teacher, cached, teacher_s = get_or_train_teacher(cfg, x_train, y_train, use_cache=not args.no_teacher_cache)
    teacher_acc = test_accuracy(teacher, x_test, y_test)
    print(f"[teacher] test accuracy = {100 * teacher_acc:.2f}%  params={teacher.count_params():,}")

    # ---- baseline student: labels only --------------------------------------
    tb = cfg["training"]["baseline_student"]
    base = build_student(cfg, "baseline_student")
    compile_supervised(base, float(tb["learning_rate"]))
    t0 = time.time()
    base.fit(x_train, y_train, epochs=args.epochs or int(tb["epochs"]), batch_size=int(tb["batch_size"]),
             validation_split=vs, verbose=2)
    base_s = time.time() - t0
    base_acc = test_accuracy(base, x_test, y_test)
    print(f"[base ] baseline student test accuracy = {100 * base_acc:.2f}%  ({base_s:.1f}s)")

    # ---- knowledge-distilled student ---------------------------------------
    tk = cfg["training"]["kd_student"]
    student = build_student(cfg, "kd_student")
    distiller = Distiller(student, teacher, alpha=float(dcfg["alpha"]), temperature=float(dcfg["temperature"]))
    distiller.compile(optimizer=keras.optimizers.Adam(learning_rate=float(tk["learning_rate"])), metrics=["accuracy"])
    t0 = time.time()
    distiller.fit(x_train, y_train, epochs=args.epochs or int(tk["epochs"]), batch_size=int(tk["batch_size"]),
                  validation_split=vs, verbose=2)
    kd_s = time.time() - t0
    kd_acc = test_accuracy(student, x_test, y_test)
    print(f"[kd   ] KD student test accuracy = {100 * kd_acc:.2f}%  "
          f"(KD gain vs baseline student {100 * (kd_acc - base_acc):+.2f} pp, {kd_s:.1f}s)")

    params = {
        "model_source": "pipeline_train",
        "teacher_params": int(teacher.count_params()),
        "student_params": int(student.count_params()),
        "teacher_epochs": cfg["training"]["teacher"]["epochs"],
        "student_epochs": args.epochs or tk["epochs"],
        "batch_size": tk["batch_size"],
        "kd_alpha": dcfg["alpha"],
        "kd_temperature": dcfg["temperature"],
        "teacher_from_cache": cached,
        "teacher_cache_key": teacher_cache_key(cfg),
    }
    metrics = {
        "teacher_keras_test_accuracy": teacher_acc,
        "baseline_student_keras_test_accuracy": base_acc,
        "kd_student_keras_test_accuracy": kd_acc,
        "kd_gain_vs_baseline_student": kd_acc - base_acc,
        "teacher_train_seconds": round(teacher_s, 2),
        "baseline_student_train_seconds": round(base_s, 2),
        "kd_student_train_seconds": round(kd_s, 2),
    }
    return student, params, metrics


def import_mode(cfg, args, data) -> Tuple[keras.Model, Dict[str, Any], Dict[str, Any]]:
    (_, _), (x_test, y_test) = data
    h5 = Path(args.import_h5)
    # Build the architecture from config.yaml and load only the weights: this is robust to
    # Keras version differences in the serialized layer configs (Colab's newer tf_keras writes
    # fields TF 2.16 cannot parse) and it verifies the imported model really IS the configured
    # student - any layer/shape mismatch raises here.
    student = build_student(cfg, "kd_student")
    try:
        student.load_weights(h5)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[import] {h5} does not match models.student in config.yaml: {exc}") from exc
    acc = test_accuracy(student, x_test, y_test)
    print(f"[import] {h5.name}: {student.count_params():,} params, Keras test accuracy {100 * acc:.2f}%")
    params = {"model_source": "colab_import", "imported_h5": h5.name, "imported_h5_sha256": sha256(h5.read_bytes()),
              "student_params": int(student.count_params())}
    if args.import_tflite:
        params["imported_tflite"] = Path(args.import_tflite).name
    metrics = {"kd_student_keras_test_accuracy": acc}
    return student, params, metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--epochs", type=int, default=None, help="override student epochs (baseline + KD)")
    ap.add_argument("--no-teacher-cache", action="store_true", help="always retrain the teacher")
    ap.add_argument("--import-h5", help="IMPORT mode: trained KD student Keras .h5 (e.g. from Colab)")
    ap.add_argument("--import-tflite", help="IMPORT mode: ship this exact INT8 .tflite instead of re-quantizing")
    args = ap.parse_args(argv)
    if args.import_tflite and not args.import_h5:
        ap.error("--import-tflite needs --import-h5 (the FP32 reference for the accuracy-drop gate)")

    cfg = load_config(args.config)
    set_seeds(int(cfg["project"]["seed"]))
    resolve(cfg["paths"]["artifacts_dir"]).mkdir(parents=True, exist_ok=True)
    fp32_path, int8_path = resolve(cfg["paths"]["fp32_tflite"]), resolve(cfg["paths"]["int8_tflite"])
    metrics_path = resolve(cfg["paths"]["metrics_json"])

    mode = "IMPORT (Colab KD student)" if args.import_h5 else "TRAIN (teacher -> KD student)"
    print("=" * 72)
    print(f" TinyMLOps | Stage 1: knowledge distillation + INT8 PTQ | mode: {mode}")
    print(f" TensorFlow {tf.__version__} | Python {platform.python_version()}")
    print("=" * 72)

    data = load_mnist()
    (x_train, _), _ = data
    print(f"[data] train={x_train.shape} test={data[1][0].shape} dtype={x_train.dtype}")

    student, params, metrics = (import_mode if args.import_h5 else train_mode)(cfg, args, data)
    student.summary(print_fn=lambda s: print("   " + s))

    fp32_bytes = convert_fp32(student)
    fp32_path.write_bytes(fp32_bytes)
    if args.import_tflite:
        int8_bytes = Path(args.import_tflite).read_bytes()
        print(f"[int8] using the imported INT8 model byte-for-byte: {args.import_tflite}")
    else:
        rep = make_representative_dataset(x_train, int(cfg["quantization"]["representative_samples"]))
        int8_bytes = convert_int8_ptq(student, rep)
    int8_path.write_bytes(int8_bytes)
    print(f"[fp32] {_rel(fp32_path)}  {len(fp32_bytes):,} B")
    print(f"[int8] {_rel(int8_path)}  {len(int8_bytes):,} B  "
          f"({len(fp32_bytes) / len(int8_bytes):.2f}x smaller)")

    params.update({"quantization": cfg["quantization"]["method"],
                   "representative_samples": cfg["quantization"]["representative_samples"],
                   "seed": cfg["project"]["seed"], "tf_version": tf.__version__})
    metrics.update({"fp32_tflite_bytes": len(fp32_bytes), "int8_tflite_bytes": len(int8_bytes),
                    "compression_ratio": round(len(fp32_bytes) / len(int8_bytes), 3)})
    run_id = log_to_mlflow(cfg, params, metrics, [fp32_path, int8_path],
                           "colab-import" if args.import_h5 else "kd-train")
    record = {"params": params, "metrics": metrics, "fp32_sha256": sha256(fp32_bytes),
              "int8_sha256": sha256(int8_bytes), "mlflow_run_id": run_id,
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    metrics_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[done] metrics -> {_rel(metrics_path)}  mlflow_run_id={run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
