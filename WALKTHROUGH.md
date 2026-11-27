# Live Demonstration & Verification Walkthrough

**CDS 303 – TinyML · TinyMLOps MNIST on ESP32-WROOM-32**

The demo has six checkpoints. Each one lists the commands to run, what the audience should see, and a **✔ Verify** line saying what counts as success. The whole walkthrough takes about 20–25 minutes. Training takes about 2–4 minutes on a laptop CPU, so start it early (see the tip under Checkpoint 2).

> Every command is run from the repository root. Windows users without `make` can use the raw Python, `pio` and `wokwi-cli` command shown under each step.

---

## Checkpoint 1 — Environment and versioning setup

**Goal:** show that the toolchain is pinned and reproducible.

```bash
python --version                      # 3.11.x or 3.12.x
pip install -r requirements.txt       # first run only
python -c "import tensorflow as tf, tensorflow_model_optimization as t; print('TF', tf.__version__, '| tfmot', t.__version__)"
pio --version                         # PlatformIO Core 6.1.x
wokwi-cli --version                   # Wokwi CLI
dvc --version                         # 3.55.x
mlflow --version                      # 2.16.x
dvc dag                               # prepare_train -> evaluate_gate -> codegen_cpp
```

Talking points:
* `ml_pipeline/config.yaml` is the only place hyperparameters, quantization settings, gate thresholds and the hardware budget are defined. Every script reads it.
* `dvc.yaml` declares what each stage depends on and produces, so `dvc repro` reruns only the stages affected by a change.

**✔ Verify:** every tool prints a version. `dvc dag` shows the three stages.

---

## Checkpoint 2 — FP32 baseline vs. QAT training with MLflow tracking

```bash
make train            # or: python ml_pipeline/train_and_quantize.py
make mlflow-ui        # or: mlflow ui --backend-store-uri ./mlruns  → http://127.0.0.1:5000
```

What to show:
1. The console prints the Keras summary: Conv2D(12) → MaxPool → Flatten → Dense(10), 20,410 parameters.
2. The FP32 model trains for 2 epochs (the reference run reached 97.31% FP32 test accuracy; QAT fine-tuning raised it to 97.83%). The model is then wrapped with `tfmot.quantization.keras.quantize_model` and fine-tuned for 2 more epochs. That is quantization-aware training: fake-quant nodes learn the INT8 ranges during training.
3. Conversion writes `artifacts/baseline_fp32.tflite` (83,836 B in the reference run) and `artifacts/model_qat_int8.tflite` (23,432 B), 3.58× smaller.
4. In MLflow, open experiment **tinymlops-mnist-esp32** and the latest run. Show the params (epochs, learning rates, seed), the metrics (`fp32_keras_test_accuracy`, `qat_keras_test_accuracy`, `int8_tflite_bytes`, `compression_ratio`) and the two `.tflite` files under Artifacts. After Checkpoint 3 the same run also has the gate metrics and the tag `deployment_gate=PASSED`.

Talking point: `TF_USE_LEGACY_KERAS=1` is set before TensorFlow is imported, so `tf.keras` is Keras 2 (`tf_keras`). tfmot's QAT API doesn't support Keras 3.

**✔ Verify:** `artifacts/metrics.json` exists and the MLflow run shows both accuracies.

> Tip: start `make train` while you present Checkpoint 1.

---

## Checkpoint 3 — Automated quality gates and failure simulation

```bash
make gate             # or: python ml_pipeline/evaluate_and_gate.py
```

Expected output (your numbers will differ slightly):

```
[eval] 10,000 images in 2.3s | FP32=97.31% INT8=97.83% | FP32/INT8 top-1 agreement=98.92%
+------------------+--------+------------------------------------------------+------------------------------------------+
| GATE             | STATUS | MEASURED                                       | THRESHOLD                                |
| INT8_IO_QUANT    | PASS   | in int8 s=0.00392157 zp=-128 | out int8 ...    | int8 in/out, scale>0, zp in [-128,127]   |
| FLASH_SIZE       | PASS   | 23,432 B                                       | <= 35,000 B                              |
| OPERATOR_AUDIT   | PASS   | CONV_2D, MAX_POOL_2D, RESHAPE, FULLY_CONNECTED | subset of CONV_2D, MAX_POOL_2D, ...      |
| INT8_ACCURACY    | PASS   | 97.83%                                         | >= 96.00%                                |
| ACCURACY_DROP    | PASS   | -0.52 pp                                       | <= 1.50 pp                               |
[golden] exported 10 vectors ...
[GATE] ALL QUALITY GATES PASSED -> model approved for ESP32 deployment
```

**Failure simulation.** Tighten the flash ceiling to 10 KB:

```bash
make gate-fail-demo
# or: python ml_pipeline/evaluate_and_gate.py --max-bytes 10000 --report-out artifacts/gate_report_demo.json --golden-out artifacts/golden_vectors_demo.json --no-mlflow ; echo "exit=$?"
```

You will see `FLASH_SIZE | FAIL`, then `[GATE] DEPLOYMENT BLOCKED -> failed gate(s): FLASH_SIZE`, then `exit=1`. In CI that non-zero exit code stops the job, so no firmware is built and nothing is simulated. The demo writes to separate `*_demo.json` files, so the approved model's report is left alone.

Other blocks you can show:
* `--min-accuracy 0.995` makes the accuracy floor fail.
* `--max-drop 0.0` makes the accuracy-drop budget fail.
* Show `test_operator_gate_allows_only_micro_ops` in `tests/test_ml_pipeline.py`: a model containing `SOFTMAX` or a custom op is rejected, because those kernels aren't registered in the firmware.

Talking point: the operator audit parses the `.tflite` flatbuffer itself, reading `Model.operator_codes` (see `parse_tflite_opcodes`). It needs neither TensorFlow nor private APIs.

**✔ Verify:** the normal gate exits 0. The demo prints `DEPLOYMENT BLOCKED` and exits 1.

---

## Checkpoint 4 — C++ flatbuffer codegen and firmware compilation

```bash
make build-fw         # = make codegen + pio run -d firmware_esp32 + size report
# raw: python ml_pipeline/export_to_cpp.py && pio run -d firmware_esp32
```

What to show:
1. The top of `firmware_esp32/src/model_data.h` has the SHA-256, timestamp, accuracy, operator list and input/output scale and zero-point, followed by `alignas(16) const unsigned char g_mnist_model_data[] = {...}`. TFLM requires 16-byte alignment for its tensor buffers.
2. `firmware_esp32/src/test_vectors.h` has `kTestImages[10][784]` (already INT8), `kTestLabels`, `kExpectedLogits` (the host reference outputs) and `kInputScale/ZeroPoint`, `kOutputScale/ZeroPoint`.
3. In `main.cpp`, `MicroMutableOpResolver<6>` registers exactly six kernels (not `AllOpsResolver`), and `kTensorArenaSize = 32 * 1024`.
4. PlatformIO compiles for Xtensa LX6 and prints:

```
RAM:   [==        ]  18.2% (used 59680 bytes from 327680 bytes)
Flash: [===       ]  28.5% (used 373853 bytes from 1310720 bytes)

  ESP32-WROOM-32 budget
  Flash :    373853 B =   8.91% of 4 MB
  SRAM  :     59680 B =  11.21% of 520 KB (static DRAM incl. 32 KB tensor arena)
```

Talking point: the whole application, including the Arduino core, WiFi-capable IDF, TFLM and the model, uses less than 10% of the 4 MB flash. The model itself is 23,432 B.

**✔ Verify:** `pio run` ends with `SUCCESS`, and `firmware_esp32/.pio/build/esp32dev/firmware.bin` and `firmware.elf` exist.

---

## Checkpoint 5 — Wokwi ESP32 simulation (BIST and interactive UART streaming)

> **Rehearse this checkpoint before the demo.** The Wokwi stage was not run while this repository was being built, because no `WOKWI_CLI_TOKEN` was available. The output below is what the firmware is written to print. Do one dry run of `make sim` with your token first, to confirm the scenario steps (`write-serial`, `expect-pin`) pass.

### 5a. Headless simulation (wokwi-cli)

```bash
export WOKWI_CLI_TOKEN=wok_...         # PowerShell: $env:WOKWI_CLI_TOKEN="wok_..."
make sim
# raw: cd firmware_esp32 && wokwi-cli --timeout 120000 --scenario test_scenario.yaml --serial-log-file ../artifacts/sim_serial.log --fail-text "Guru Meditation" .
```

What the virtual ESP32 prints:

```
[TELEMETRY] MODEL_SHA256_EMBEDDED=3f9c...   MODEL_SHA256_RUNTIME =3f9c...
[TELEMETRY] MODEL_INTEGRITY=OK
[TELEMETRY] HEAP_TOTAL=... HEAP_FREE_BEFORE_ALLOC=... HEAP_FREE_AFTER_ALLOC=...
[TELEMETRY] ARENA_USED=1xxxx ARENA_SIZE=32768 ARENA_UTIL=..%
[BIST] Built-in self test: 10 golden MNIST digits (INT8)
| Digit | Expected | Predicted | Confidence | Latency (us) | Parity | Status |
|     0 |        0 |         0 |     99.9x% |         xxxx | EXACT  | PASS   |
...
[TINYMLOPS_PARITY] BITEXACT=10/10
[TINYMLOPS_TEST_SUMMARY] PASSED=10/10 ACCURACY=100.00% AVG_LATENCY_US=... ARENA_BYTES=...
[STATUS] BIST PASSED - status LED (GPIO2) ON
```

The scenario `test_scenario.yaml` then checks that:
* GPIO2 (the green LED) is high,
* `PING` returns `PONG`,
* `SELFTEST:7` and `SELFTEST:3` return the correct prediction over UART,
* and the run fails immediately on `Guru Meditation`, or by timeout if `[TINYMLOPS_FATAL]` appears (for example when `AllocateTensors()` fails).

`artifacts/sim_report.md` is generated from the log by `serial_hil_verifier.py --parse-log`.

**Failure demo (optional):** set `-DTINYMLOPS_TENSOR_ARENA_KB=4` in `platformio.ini`, then run `make build-fw sim`. The firmware prints `[TINYMLOPS_FATAL] AllocateTensors() failed` and the scenario fails. Change the value back afterwards.

### 5b. Interactive: watch it in VS Code and stream live MNIST frames

1. Install the **Wokwi Simulator** VS Code extension, open `firmware_esp32/`, press F1, and choose *Wokwi: Start Simulator*. The diagram shows the ESP32 DevKit, and the green LED blinks five times and then stays on.
2. `wokwi.toml` exposes the simulated UART as an RFC2217 server on port 4000. In a second terminal:

```bash
make hil PORT=rfc2217://localhost:4000 EXTRA=50
# raw: python tests/serial_hil_verifier.py --port rfc2217://localhost:4000 --extra 50
```

The verifier sends each golden digit, plus 50 random MNIST test images, as `INFER:<784 int8>`. It parses the JSON reply (`{"pred":..,"score":..,"latency_us":..,"logits":[..]}`) and compares all 10 INT8 logits **bit for bit** with the Python `tf.lite.Interpreter`:

```
| golden[7]    |     7 |    7 |    7 |        xxxx | EXACT  |
[HIL_SUMMARY] SAMPLES=60 BITEXACT=60/60 DEVICE_ACCURACY=9x.xx% AVG_LATENCY_US=... BUDGET_US=100000
[HIL] PASS
```

The same command works with a physical board: `make flash`, then `make hil PORT=COM5` (or `/dev/ttyUSB0`).

Without the simulator or a board, you can still show the host side of the parity check:

```bash
make host-parity      # python tests/serial_hil_verifier.py --host-only
```

**✔ Verify:** `PASSED=10/10`, `BITEXACT=10/10`, `MODEL_INTEGRITY=OK`, LED on, and `[HIL] PASS`.

---

## Checkpoint 6 — End-to-end CI/CD in GitHub Actions

```bash
# 1. Change a hyperparameter
sed -i 's/conv_filters: 12/conv_filters: 8/' ml_pipeline/config.yaml   # or edit it by hand
dvc params diff                                                       # shows exactly what changed
# 2. Push
git checkout -b demo/filters-8 && git commit -am "exp: 8 conv filters" && git push -u origin demo/filters-8
# 3. Open a PR to main, or run the workflow manually: Actions → TinyMLOps MNIST ESP32 Pipeline → Run workflow
```

What to show in the Actions run:
1. **Job 1 · Train, quantize, gate, codegen** runs training, the gates, codegen, pytest and the host parity check, then uploads `tinyml-model` and `firmware-headers` as artifacts.
2. **Job 2 · ESP32 build + Wokwi HIL simulation** downloads the new headers, runs `pio run`, boots the firmware in Wokwi with `wokwi/wokwi-ci-action@v1`, and parses the serial log.
3. The **Summary** tab shows three generated tables:
   * *Model Training & Quality Gates*: FP32 vs. INT8 accuracy and size, plus the gate table
   * *ESP32-WROOM-32 Firmware Footprint*: flash as % of 4 MB, SRAM as % of 520 KB
   * *TinyMLOps Deployment & Simulation Report*: the BIST table, latency, arena, parity and integrity
4. To show a blocked pipeline, push `max_model_bytes: 10000` in `config.yaml`. Job 1 fails at the gate step and Job 2 never runs.

**✔ Verify:** both jobs are green and the Step Summary contains all three report tables.

---

## Appendix — Q&A cheat sheet

| Likely question | Short answer |
|---|---|
| Why QAT and not post-training quantization? | QAT simulates INT8 rounding during fine-tuning, so the weights adapt to it. The drop is typically under 0.1 pp. |
| Why `int8` input and output? | The MCU never touches floats on the inference path. The host quantizes images with the model's input scale (≈1/255) and zero-point (−128). |
| How do you know the device computes the same thing as Python? | TFLM and `BUILTIN_REF` use identical integer kernels. The host logits are embedded in the firmware and compared bit for bit (`BITEXACT=10/10`), and the HIL verifier does the same for arbitrary images. |
| What stops a corrupted or mismatched model being flashed? | The SHA-256 chain: `metrics.json` → `golden_vectors.json` → `model_data.h` / `test_vectors.h` → recomputed on the device at boot (`MODEL_INTEGRITY`). |
| Why 32 KB arena? | The activations need only about 11 KB (input 784 B + conv output 8,112 B + pool output 2,028 B), plus TFLM bookkeeping. Read the exact figure from `ARENA_USED` at boot. The rest is headroom for model changes, and a failed allocation is caught as `[TINYMLOPS_FATAL]`. |
| What's next? | Phase 2 adds a feature on top of this scaffold, for example A/B model hot-swapping from LittleFS, drift-triggered retraining, or dual-model switching for intermittent power. |
