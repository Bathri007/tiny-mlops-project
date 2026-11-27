# TinyMLOps: MNIST CNN on the ESP32-WROOM-32

**CDS 303 – TinyML · course project, phase 1**

This repository takes a handwritten-digit CNN through the whole TinyMLOps lifecycle. It trains the model, quantizes it to INT8, checks it against deployment gates, turns it into C++ for an ESP32-WROOM-32 (4 MB flash, 520 KB SRAM), compiles that firmware, runs it on a simulated ESP32 and writes a report. Each step is one command, and GitHub Actions runs all of them on every push.

```
 config.yaml ──► train_and_quantize.py ──► evaluate_and_gate.py ──► export_to_cpp.py ──► pio run ──► Wokwi ESP32 ──► Step Summary
   (params)     FP32 + QAT INT8 .tflite    Flash / Acc / Ops / Quant   model_data.h        Xtensa     BIST 10/10       report
                MLflow + metrics.json      gates, golden vectors       test_vectors.h      LX6 .bin   UART protocol
                        └──────────────── DVC: prepare_train → evaluate_gate → codegen_cpp ────────────────┘
```

---

## 1. What runs where

| Layer | What it checks | Where | Command |
|---|---|---|---|
| **0 · Quality gates** | INT8 model ≤ 35,000 B; INT8 accuracy ≥ 96.0% on all 10k test images; drop vs. FP32 ≤ 1.5 pp; only TFLM ops; int8 I/O quantization | host / CI | `make gate` |
| **1 · Unit tests** | preprocessing, architecture and parameter count, gate rejection logic, flatbuffer opcode parser, C++ header syntax, round trip and SHA-256, `g++` compile of the headers | host / CI | `make test` |
| **2 · Reference parity** | the INT8 logits stored in `test_vectors.h` match `tf.lite.Interpreter` (reference kernels) bit for bit; with a device attached, the device logits also match the host bit for bit | host / CI / device | `make host-parity`, `make hil` |
| **3 · HIL simulation** | the virtual ESP32 boots, allocates the 32 KB arena, verifies the model's SHA-256, passes the 10/10 BIST, turns the GPIO2 LED on and answers UART commands | Wokwi CLI / CI | `make sim` |
| **4 · CI/CD** | runs layers 0–3 on every push or PR to `main` and posts the results to the GitHub Step Summary | GitHub Actions | `git push` |

## 2. Model and hardware budget

| | |
|---|---|
| Network | `Input(28,28,1) → Conv2D(12, 3×3, ReLU) → MaxPool(2×2) → Flatten → Dense(10)` — 20,410 parameters |
| Training | FP32 baseline: 2 epochs, batch 32, validation split 0.1 → QAT fine-tune (tfmot `quantize_model`): 2 epochs |
| Conversion | Full-integer INT8: `TFLITE_BUILTINS_INT8`, `int8` input and output, 100-image representative dataset |
| Operators | `CONV_2D, MAX_POOL_2D, RESHAPE, FULLY_CONNECTED` (`QUANTIZE`/`DEQUANTIZE` are also registered) |
| Runtime | TFLite Micro with `MicroMutableOpResolver<6>` (no `AllOpsResolver`) and a 32 KB static tensor arena |
| Target | ESP32-WROOM-32, Xtensa LX6 @ 240 MHz, Arduino-ESP32 2.0.17 through PlatformIO |

Actual numbers from a local run of this repository are in [§6](#6-reference-results).

## 3. Repository layout

```
.github/workflows/tinymlops_pipeline.yml   CI/CD: train → gate → codegen → pytest → pio build → Wokwi → report
ml_pipeline/config.yaml                    single source of truth: hyperparameters, quantization, gates, HW budget
ml_pipeline/train_and_quantize.py          FP32 + QAT INT8 training, TFLite export, MLflow logging
ml_pipeline/evaluate_and_gate.py           quality gates, dependency-free flatbuffer op audit, golden vectors
ml_pipeline/export_to_cpp.py               .tflite → alignas(16) model_data.h, golden JSON → test_vectors.h
firmware_esp32/platformio.ini              esp32dev / Arduino / TensorFlowLite_ESP32
firmware_esp32/src/main.cpp                TFLM engine, telemetry, SHA-256 integrity check, BIST, UART protocol
firmware_esp32/src/model_data.h            generated: INT8 flatbuffer + metadata
firmware_esp32/src/test_vectors.h          generated: 10 INT8 digits, labels, reference logits, quant params
firmware_esp32/wokwi.toml                  simulator config (firmware.bin / firmware.elf, RFC2217 port 4000)
firmware_esp32/diagram.json                ESP32 DevKit V1 + green status LED on GPIO2 + serial monitor
firmware_esp32/test_scenario.yaml          Wokwi automation: BIST assertion, LED pin check, UART commands
tests/test_ml_pipeline.py                  pytest suite (layer 1)
tests/serial_hil_verifier.py               host/device parity, live streaming, simulator log → Markdown report
Makefile · dvc.yaml · requirements.txt     one-command workflow, DVC stages, pinned dependencies
```

## 4. Setup

Requirements: Python 3.11 or 3.12 (TensorFlow 2.16 does not support 3.13+), git, and optionally GNU make.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt          # TF 2.16.2, tf-keras, tfmot 0.8.0, MLflow, DVC, PlatformIO, pytest

# Wokwi CLI (simulator)
curl -L https://wokwi.com/ci/install.sh | sh                 # Linux/macOS
iwr https://wokwi.com/ci/install.ps1 -useb | iex             # Windows PowerShell
export WOKWI_CLI_TOKEN=wok_xxxxxxxx                           # from https://wokwi.com/dashboard/ci
```

For GitHub Actions, add `WOKWI_CLI_TOKEN` under **Settings → Secrets and variables → Actions**. If the secret is missing, CI still trains, gates and compiles; it skips only the simulation step and notes that in the summary.

## 5. Commands

| Command | What it does |
|---|---|
| `make train` | Trains the FP32 baseline and the QAT INT8 model, writes `artifacts/*.tflite` and `metrics.json`, and logs the run to MLflow |
| `make gate` | Runs every quality gate on the full 10k test set and exports the golden vectors (exits 1 if any gate fails) |
| `make gate-fail-demo` | Lowers the flash ceiling to 10,000 B to show the pipeline blocking the model |
| `make codegen` | Generates `model_data.h` and `test_vectors.h` |
| `make build-fw` | Runs codegen and `pio run`, then prints flash as % of 4 MB and SRAM as % of 520 KB |
| `make sim` | Boots the firmware in Wokwi, runs the scenario and writes `artifacts/sim_report.md` |
| `make hil PORT=COM5 EXTRA=200` | Streams frames to a real board, or to `rfc2217://localhost:4000` from the VS Code Wokwi extension |
| `make test` / `make host-parity` | Runs the pytest suite / the host reference parity check |
| `make test-all` | Runs everything above in order (the simulator step only if `WOKWI_CLI_TOKEN` is set) |
| `make mlflow-ui` | Opens MLflow at http://127.0.0.1:5000 |
| `dvc repro` / `dvc metrics show` | Reruns only the stages whose dependencies changed / shows the versioned metrics |

Without `make` (for example in plain Windows PowerShell), run the Python scripts directly:

```powershell
python ml_pipeline/train_and_quantize.py
python ml_pipeline/evaluate_and_gate.py
python ml_pipeline/export_to_cpp.py
python -m pytest -v tests/
pio run -d firmware_esp32
cd firmware_esp32; wokwi-cli --timeout 120000 --scenario test_scenario.yaml --serial-log-file ../artifacts/sim_serial.log --fail-text "Guru Meditation" .
```

## 6. Reference results

These numbers come from a local run on Windows 11 with Python 3.12.10, TF 2.16.2 and tfmot 0.8.0 on CPU. Retraining moves the accuracy slightly, because weight initialization and data shuffling are not bit-deterministic across machines. Your own run writes its numbers to `artifacts/metrics.json` and `artifacts/gate_report.json`.

| Metric | FP32 baseline | QAT INT8 |
|---|---|---|
| TFLite test accuracy (all 10,000 images) | 97.31% | **97.83%** |
| `.tflite` size | 83,836 B | **23,432 B** (3.58× smaller, 0.56% of 4 MB flash) |
| Accuracy drop FP32 → INT8 | — | −0.52 pp (QAT came out *better*; budget ≤ 1.5 pp) |
| FP32 / INT8 top-1 agreement | — | 98.92% |
| Operators | — | `CONV_2D, MAX_POOL_2D, RESHAPE, FULLY_CONNECTED` |
| Input / output quantization | — | int8, s = 0.00392157, zp = −128 / int8, s ≈ 0.12 |

| Check | Result |
|---|---|
| Quality gates (`make gate`) | all 5 PASS, exit 0 |
| Gate failure demo (`--max-bytes 10000`) | `FLASH_SIZE FAIL`, `DEPLOYMENT BLOCKED`, exit 1 |
| Golden vectors | MNIST test indices 3, 2, 1, 30, 4, 8, 11, 0, 61, 7 (digits 0–9) |
| Host reference parity (`make host-parity`) | 10/10 EXACT |
| pytest (`make test`) | 20 passed |
| Firmware build (`pio run`, espressif32 6.9.0, TensorFlowLite_ESP32 1.0.0) | SUCCESS, `firmware.bin` 374,224 B |
| Firmware flash | 373,853 B = 8.91% of 4 MB (28.5% of the 1.25 MB app partition) |
| Firmware static RAM | 59,680 B = 11.21% of 520 KB (18.2% of 320 KB DRAM), including the 32 KB tensor arena |

**Not yet verified locally:** `make sim` and `make hil` (Wokwi). They need a `WOKWI_CLI_TOKEN`, which this machine didn't have. That means the on-device BIST output, arena usage, latency, bit-exact device parity, and the `test_scenario.yaml` steps (`write-serial` and the GPIO2 `expect-pin` check) have not been run yet. They run in CI once the secret is added, or locally with the token set.

## 7. Firmware serial protocol (115200 8N1, newline-terminated)

| Command | Response |
|---|---|
| `INFER:<784 int8 comma-separated>` or `INFER:<1568 hex chars>` | `{"pred":7,"score":118,"latency_us":1420,"confidence":99.71,"logits":[...]}` |
| `SELFTEST:<0-9>` | Same JSON plus `idx`, `label`, and `parity` (`EXACT`/`DIFF` against the host reference logits) |
| `BIST` / `INFO` / `PING` | Reruns the self-test / reprints telemetry / `PONG` |

Boot output includes machine-parseable lines:

```
[TELEMETRY] MODEL_SHA256_EMBEDDED=<sha>      [TELEMETRY] MODEL_INTEGRITY=OK
[TELEMETRY] HEAP_TOTAL=.. HEAP_FREE_BEFORE_ALLOC=.. HEAP_FREE_AFTER_ALLOC=..
[TELEMETRY] ARENA_USED=.. ARENA_SIZE=32768 ARENA_UTIL=..%
[TINYMLOPS_PARITY] BITEXACT=10/10
[TINYMLOPS_TEST_SUMMARY] PASSED=10/10 ACCURACY=100.00% AVG_LATENCY_US=<val> ARENA_BYTES=<val>
```

## 8. Design decisions

* **The gates are the deployment contract.** `export_to_cpp.py` refuses to generate code if `gate_report.json` shows a failure or if the golden vectors' SHA-256 doesn't match the model. A model that failed a gate therefore can't reach the firmware, either by mistake or by hand.
* **The operator audit has no dependencies.** The `.tflite` flatbuffer is parsed directly (`Model.operator_codes`, including the `deprecated_builtin_code` compatibility rule), so the audit doesn't rely on private TF APIs and can be unit-tested on synthetic flatbuffers.
* **Parity is bit-exact.** The host uses `OpResolverType.BUILTIN_REF`, the same integer reference kernels as TFLite Micro. Its INT8 logits are embedded in the firmware, and the device compares its own logits against them on every boot.
* **The firmware checks the model's integrity.** It recomputes the flatbuffer's SHA-256 in software at boot and compares it with the hash the pipeline stamped into `model_data.h` and `test_vectors.h`.
* **The resolver is minimal.** `MicroMutableOpResolver<6>` links only six kernels, so the binary is smaller than with `AllOpsResolver`.

See **[WALKTHROUGH.md](WALKTHROUGH.md)** for the six-checkpoint live demo script.
