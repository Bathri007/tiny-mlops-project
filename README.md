# TinyMLOps: Knowledge-Distilled MNIST Student on the ESP32-WROOM-32

**CDS 303 – TinyML · course project**

This project deploys a **knowledge-distilled student CNN** for handwritten-digit recognition on an ESP32-WROOM-32 (4 MB flash, 520 KB SRAM), and automates the whole lifecycle around it:

1. A large **teacher** CNN (503,690 parameters) trains on MNIST.
2. A tiny **student** (14,410 parameters, 35× smaller) learns from the teacher's softened predictions as well as from the labels.
3. The student is **quantized to full INT8** and checked against deployment gates.
4. It is turned into C++ and compiled into ESP32 firmware.
5. The firmware is **run on a simulated ESP32** and checked bit for bit against the host.

GitHub Actions runs all of this on every push. The work builds on CDS 303 Assignment 4 (knowledge distillation). The exact Colab-trained model from that assignment can be pushed through the same pipeline.

```
 config.yaml ─► train_and_quantize.py ─────────────► evaluate_and_gate.py ─► export_to_cpp.py ─► pio run ─► Wokwi ESP32 ─► Step Summary
  (params)     teacher (cached) → baseline student     Flash / Acc / Ops /    model_data.h       Xtensa     BIST 10/10      report
               → KD student → INT8 PTQ .tflite         Quant gates, golden    test_vectors.h     LX6 .bin   UART protocol
               MLflow + metrics.json                   vectors
     or: --import-h5/--import-tflite  (exact Colab Assignment 4 model)
                    └──────────────── DVC: prepare_train → evaluate_gate → codegen_cpp ────────────────┘
```

---

## 1. The models

| | Teacher | Student (deployed) |
|---|---|---|
| Architecture | Conv32 → Conv64 → Pool → Conv128 → Pool → Dense128 → Dense10 | Conv8 → Pool → Conv16 → Pool → Dense32 → Dense10 |
| Parameters | 503,690 | **14,410** |
| Role | trains once and is cached; never deployed | trained twice: on labels only (baseline control) and by **distillation** (deployed) |

**Distillation loss**, as in Assignment 4 (`ml_pipeline/train_and_quantize.py:Distiller`):

```
loss = α · CE(y, student_logits) + (1 − α) · T² · KL( softmax(teacher/T) ‖ softmax(student/T) )      α = 0.1, T = 5
```

The student learns the hard labels and also how the teacher ranks the wrong digits (its "dark knowledge"). The deployed model is then converted with full-integer **post-training quantization**: int8 weights, activations, input and output, calibrated on the first 200 training images.

Architectures, epochs, α, T and the gates all live in `ml_pipeline/config.yaml`.

## 2. Where the deployed model comes from

| Mode | Command | Use |
|---|---|---|
| **train** (default, DVC + CI) | `make train` | The pipeline distills the student itself. The teacher is cached in `artifacts/teacher_cache/`, keyed by a hash of its configuration, so it retrains only when its own settings change. |
| **colab_import** | `make import-colab` | Ships the **exact** Assignment 4 model (`models/colab/student_kd.h5` + `student_kd_int8.tflite`, byte for byte). The architecture is rebuilt from `config.yaml` and the weights are loaded into it, so a model that isn't the configured student is rejected. In CI: **Actions → Run workflow → model_source: colab_import**. |

Both modes produce `artifacts/kd_student_fp32.tflite` and `artifacts/kd_student_int8.tflite`. Everything downstream is identical.

## 3. What runs where

| Layer | What it checks | Where | Command |
|---|---|---|---|
| **0 · Quality gates** | INT8 model ≤ 35,000 B; INT8 accuracy ≥ 96.0% on all 10k test images; drop vs. FP32 KD student ≤ 1.5 pp; only TFLite Micro ops; int8 I/O quantization | host / CI | `make gate` |
| **1 · Unit tests (23)** | teacher/student architectures and parameter counts; that the distiller updates only the student; teacher cache key; that import rejects the wrong model; gate logic; flatbuffer op parser; C++ header round trip, SHA-256 and `g++` compile | host / CI | `make test` |
| **2 · Reference parity** | INT8 logits in `test_vectors.h` equal `tf.lite.Interpreter` (reference kernels) bit for bit; on a device, device logits equal host logits bit for bit | host / CI / device | `make host-parity`, `make hil` |
| **3 · HIL simulation** | virtual ESP32 boots, verifies the model's SHA-256, allocates the 32 KB arena, passes the 10/10 BIST, turns the GPIO2 LED on, answers `PING` / `SELFTEST` | Wokwi CLI / CI | `make sim` |
| **4 · CI/CD** | layers 0–3 on every push or PR to `main`; results in the GitHub Step Summary | GitHub Actions | `git push` |

The KD comparison (teacher vs. baseline student vs. KD student) is **reported, not gated**. The KD gain on MNIST is a fraction of a point and varies between runs, so it mustn't block a deployment.

## 4. Repository layout

```
.github/workflows/tinymlops_pipeline.yml   CI/CD (train | colab_import) → gate → codegen → pytest → pio build → Wokwi → report
ml_pipeline/config.yaml                    single source of truth: architectures, KD (α, T), PTQ, gates, HW budget
ml_pipeline/train_and_quantize.py          teacher (cached) → baseline student → KD student → INT8 PTQ; import mode; MLflow
ml_pipeline/evaluate_and_gate.py           quality gates, dependency-free flatbuffer op audit, golden vectors, KD report
ml_pipeline/export_to_cpp.py               .tflite → alignas(16) model_data.h, golden JSON → test_vectors.h
models/colab/                              exact Assignment 4 KD student (Keras .h5 + INT8 .tflite)
firmware_esp32/platformio.ini              esp32dev / Arduino / TensorFlowLite_ESP32
firmware_esp32/src/main.cpp                TFLM engine, telemetry, SHA-256 integrity check, BIST, UART protocol
firmware_esp32/src/model_data.h            generated: INT8 KD student flatbuffer + metadata
firmware_esp32/src/test_vectors.h          generated: 10 INT8 digits, labels, reference logits, quant params
firmware_esp32/wokwi.toml · diagram.json   simulator config (RFC2217 port 4000) + ESP32 DevKit with green LED on GPIO2
firmware_esp32/test_scenario.yaml          Wokwi automation: BIST, LED pin check, UART commands
tests/test_ml_pipeline.py                  pytest suite (layer 1)
tests/serial_hil_verifier.py               host/device parity, live streaming, simulator log → Markdown report
Makefile · dvc.yaml · requirements.txt     one-command workflow, DVC stages, pinned dependencies
```

## 5. Setup

Requirements: Python 3.11 or 3.12 (TensorFlow 2.16 doesn't support 3.13+), git, and optionally GNU make.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt          # TF 2.16.2, tf-keras, MLflow, DVC, PlatformIO, pytest

# Wokwi CLI (simulator)
curl -L https://wokwi.com/ci/install.sh | sh                 # Linux/macOS
iwr https://wokwi.com/ci/install.ps1 -useb | iex             # Windows PowerShell
export WOKWI_CLI_TOKEN=wok_xxxxxxxx                           # from https://wokwi.com/dashboard/ci
```

Activate the venv before running `dvc repro`: DVC runs its stages with whatever `python` is first on `PATH`.

For GitHub Actions, add `WOKWI_CLI_TOKEN` under **Settings → Secrets and variables → Actions**.

## 6. Commands

| Command | What it does |
|---|---|
| `make train` | Teacher (from cache if possible) → baseline student → KD student → INT8 PTQ; writes `artifacts/` and logs to MLflow |
| `make import-colab` | Same outputs, but from the exact Colab Assignment 4 model in `models/colab/` |
| `make gate` | All quality gates on the full 10k test set; exports golden vectors (exit 1 on any failure) |
| `make gate-fail-demo` | Lowers the flash ceiling to 10,000 B and shows the pipeline blocking the model |
| `make codegen` | Generates `model_data.h` and `test_vectors.h` |
| `make build-fw` | codegen + `pio run`, then flash as % of 4 MB and SRAM as % of 520 KB |
| `make sim` | Boots the firmware in Wokwi, runs the scenario, writes `artifacts/sim_report.md` |
| `make hil PORT=COM5 EXTRA=200` | Streams frames to a real board, or to `rfc2217://localhost:4000` from the VS Code Wokwi extension |
| `make test` / `make host-parity` | pytest suite / host reference parity check |
| `make mlflow-ui` | MLflow at http://127.0.0.1:5000 (teacher, baseline and KD accuracies per run) |
| `dvc repro` / `dvc metrics show` / `dvc params diff` | rerun only the changed stages / versioned metrics / changed hyperparameters |

Without `make` (plain PowerShell), run the scripts directly:

```powershell
python ml_pipeline/train_and_quantize.py        # or: --import-h5 models/colab/student_kd.h5 --import-tflite models/colab/student_kd_int8.tflite
python ml_pipeline/evaluate_and_gate.py
python ml_pipeline/export_to_cpp.py
python -m pytest -v tests/
pio run -d firmware_esp32
```

## 7. Reference results

These numbers come from local runs on Windows 11, Python 3.12.10, TF 2.16.2, CPU. TensorFlow on a multi-threaded CPU isn't bit-deterministic, so retraining moves the accuracies by a few tenths of a point; two `train` runs gave a teacher of 99.19% and 99.33%. Your own run writes its numbers to `artifacts/metrics.json` and `artifacts/gate_report.json`.

**Knowledge distillation** (Keras FP32, all 10,000 test images):

| Model | Params | `train` mode (DVC run committed here) | Colab Assignment 4 run |
|---|---|---|---|
| Teacher | 503,690 | 99.33% | 99.21% |
| Baseline student (labels only) | 14,410 | 97.74% | 97.97% |
| **KD student** | 14,410 | **98.23%** (+0.49 pp from KD) | **98.24%** (+0.27 pp) |

**Deployed INT8 KD student** (TFLite, all 10,000 test images):

| | `train` mode | `colab_import` mode |
|---|---|---|
| INT8 accuracy | **98.25%** | **98.30%** |
| FP32 → INT8 drop | −0.02 pp | −0.06 pp |
| `.tflite` size | 18,816 B (0.45% of 4 MB) | 19,960 B (Colab's newer TF adds metadata) |
| Operators | `CONV_2D, MAX_POOL_2D, RESHAPE, FULLY_CONNECTED` | same |
| Quality gates | all 5 PASS | all 5 PASS |

| Check | Result |
|---|---|
| Gate failure demo (`--max-bytes 10000`) | `FLASH_SIZE FAIL`, `DEPLOYMENT BLOCKED`, exit 1 |
| Host reference parity (`make host-parity`) | 10/10 EXACT |
| pytest (`make test`) | 23 passed |
| Firmware build (espressif32 6.9.0, TensorFlowLite_ESP32 1.0.0) | SUCCESS |
| Firmware flash | 369,301 B = 8.80% of 4 MB (28.2% of the 1.25 MB app partition) |
| Firmware static RAM | 59,680 B = 11.21% of 520 KB, including the 32 KB tensor arena |

**Wokwi simulation in CI** (the earlier 20k-parameter CNN on the same firmware): BIST 10/10, `BITEXACT=10/10`, `MODEL_INTEGRITY=OK`, arena 11,108 / 32,768 B, LED / `PING` / `SELFTEST` all passing. Simulated latency is about 1 s per inference. Wokwi isn't cycle-accurate, so simulation is only checked against a loose 5 s ceiling. The 100 ms budget applies to `make hil` on a real board, and **real-board latency hasn't been measured yet**.

## 8. Over-the-air (WiFi) model updates

Firmware 2.0.0 updates its **model** over WiFi without reflashing. The firmware itself is not replaced. Every gate-approved model on `main` is published as a GitHub Release, and devices pick it up from `https://github.com/Bathri007/tiny-mlops-project/releases/latest/download/manifest.json`.

```
CI Job 3: package_model.py -> release model-vN: model.tflite + vectors.bin + manifest.json  (marked "latest")
ESP32:    WiFi -> NTP -> HTTPS manifest -> decide -> download into the idle slot (A/B), SHA-256 checked
          -> load it -> self-test with ITS golden vectors (10/10 correct AND bit-exact)
          -> pass: switch (/state.txt)   fail: put the previous model back (rollback)
CI Job 4: a simulated ESP32 (Wokwi, virtual WiFi) with the committed factory model must download,
          verify, self-test and activate the release Job 3 just published
```

| Safety net | What it catches |
| --- | --- |
| Pinned root CAs + real date via NTP | Downloads from anything other than GitHub's real servers |
| SHA-256 of model and vectors, checked while streaming | Corrupted or tampered files (`--demo-fault sha`) |
| On-device self-test before switching | A model that runs but gives wrong answers on this device (`--demo-fault selftest`) |
| Manifest checks: `min_firmware`, op list, size, no downgrades | Models this firmware cannot run, or old models replayed |
| Factory model in flash, slots A/B, atomic state file | A bad slot at boot falls back to factory; the board is never bricked |

- **WiFi:** copy `firmware_esp32/src/secrets.example.h` to `secrets.h` (git-ignored) and enter a 2.4 GHz network. Without it, the firmware uses Wokwi's `Wokwi-GUEST` network, which exists only in the simulator.
- **Try it in VS Code Wokwi:** start the simulator. The `[OTA]` lines show each step, and `[TINYMLOPS_OTA_RESULT] STATUS=UPDATED` ends a successful update. `make sim-ota` runs the same check headless.
- **Demonstrate the safety nets:** Actions → Run workflow → `demo_fault: sha` or `selftest`. Devices report `STATUS=REJECTED` or `STATUS=ROLLED_BACK` and keep serving the previous model. Publish a normal run afterwards so `latest` is a good model again.
- **Build flags:** `TINYMLOPS_OTA_ENABLED` (default 1), `TINYMLOPS_OTA_INTERVAL_S` (default 300), `TINYMLOPS_OTA_REQUIRE_BITEXACT` (default 1), and `TINYMLOPS_OTA_TLS_INSECURE` (default 0; debugging only).
- **Testing without hardware:** `tests/test_firmware_core.py` compiles the firmware's update logic (`firmware_esp32/lib/tinymlops_core`) with `g++`. It checks SHA-256 against `hashlib`, the vector pack against Python, every update decision, and real packaged releases, including both demo faults.

## 9. Firmware serial protocol (115200 8N1, newline-terminated)

| Command | Response |
|---|---|
| `INFER:<784 int8 comma-separated>` or `INFER:<1568 hex chars>` | `{"pred":7,"score":118,"latency_us":1420,"confidence":99.71,"logits":[...]}` |
| `SELFTEST:<0-9>` | Same JSON plus `idx`, `label`, and `parity` (`EXACT`/`DIFF` against the host reference logits) |
| `BIST` / `INFO` / `PING` | Reruns the self-test / reprints telemetry / `PONG` |
| `OTA` | Check for a new model now (the BOOT button does the same) |
| `STATUS` | Active model, slot, version, SHA-256, last OTA result, WiFi state |
| `FACTORY` | Switch back to the factory model compiled into the firmware |

Boot output includes machine-parseable lines:

```
[TELEMETRY] MODEL_NAME=kd_student_int8 (pipeline_train)
[TELEMETRY] MODEL_SHA256_EMBEDDED=<sha>      [TELEMETRY] MODEL_INTEGRITY=OK
[TELEMETRY] ARENA_USED=.. ARENA_SIZE=32768 ARENA_UTIL=..%
[TINYMLOPS_PARITY] BITEXACT=10/10
[TINYMLOPS_TEST_SUMMARY] PASSED=10/10 ACCURACY=100.00% AVG_LATENCY_US=<val> ARENA_BYTES=<val>
```

## 10. Design decisions

* **The teacher is cached, not versioned.** It's about 2 MB and never deployed. Caching it by a hash of its own config means changing α, T or the student retrains only the student, which saves about 1.5–3 minutes per run locally and in CI.
* **Import loads weights into the configured architecture.** It doesn't deserialize Colab's model config, so a newer Keras version in Colab doesn't matter, and an imported model that isn't the configured student fails loudly.
* **The gates are the deployment contract.** `export_to_cpp.py` refuses to generate code if `gate_report.json` shows a failure or if the golden vectors' SHA-256 doesn't match the model.
* **The operator audit has no dependencies.** The `.tflite` flatbuffer is parsed directly, so the audit doesn't rely on private TF APIs and is unit-tested on synthetic flatbuffers.
* **Parity is bit-exact.** The host uses `OpResolverType.BUILTIN_REF`, the same integer reference kernels as TFLite Micro. Its INT8 logits are embedded in the firmware, and the device checks against them on every boot.
* **The firmware checks the model's integrity.** It recomputes the flatbuffer's SHA-256 at boot and compares it with the hash stamped by the pipeline. The live HIL verifier also refuses to run against a device whose model hash differs from the host's.
* **The resolver is minimal.** `MicroMutableOpResolver<6>` links only the kernels the gate allows.

See **[WALKTHROUGH.md](WALKTHROUGH.md)** for the six-checkpoint live demo script.
