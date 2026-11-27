# Live Demonstration & Verification Walkthrough

**CDS 303 – TinyML · Knowledge-Distilled MNIST Student on ESP32-WROOM-32**

The story in one sentence: *a 504k-parameter teacher teaches a 14k-parameter student, the student is quantized to INT8, it must pass automated deployment gates, and then it runs on a (simulated) ESP32 with outputs identical bit for bit to Python.*

The demo has six checkpoints. Each one lists the commands, what the audience should see, and a **✔ Verify** line saying what counts as success. The whole walkthrough takes about 20–25 minutes. The first `make train` takes about 3.5 minutes on a laptop CPU, mostly training the teacher; after that the teacher comes from cache and a run takes about 1–1.5 minutes. Start it early (see the tip under Checkpoint 2).

> Every command is run from the repository root with the venv activated. Windows users without `make` can use the raw command shown under each step.

---

## Checkpoint 1 — Environment and versioning setup

**Goal:** show that the toolchain is pinned and reproducible.

```bash
python --version                      # 3.11.x or 3.12.x
pip install -r requirements.txt       # first run only
python -c "import tensorflow as tf; print('TF', tf.__version__)"
pio --version                         # PlatformIO Core 6.1.x
wokwi-cli --version                   # Wokwi CLI
dvc --version                         # 3.55.x
mlflow --version                      # 2.16.x
dvc dag                               # prepare_train -> evaluate_gate -> codegen_cpp
```

Talking points:
* `ml_pipeline/config.yaml` is the single source of truth: the teacher and student architectures (as layer lists), the KD hyperparameters (α = 0.1, T = 5), PTQ settings, gate thresholds and the hardware budget.
* `dvc.yaml` declares each stage's dependencies and outputs, so `dvc repro` reruns only what a change affects.

**✔ Verify:** every tool prints a version, and `dvc dag` shows the three stages.

---

## Checkpoint 2 — Knowledge distillation with MLflow tracking

```bash
make train            # or: python ml_pipeline/train_and_quantize.py
make mlflow-ui        # or: mlflow ui --backend-store-uri ./mlruns  → http://127.0.0.1:5000
```

What to show:
1. **Teacher.** The console reports `[teacher] cache hit` or trains it: 503,690 parameters, 5 epochs, about **99.3%** test accuracy. It's cached in `artifacts/teacher_cache/` under a hash of its own config, so changing α, T or the student never retrains it.
2. **Baseline student.** The 14,410-parameter student is trained on labels only. This is the control: about **97.7%**.
3. **KD student.** The same architecture is trained by the `Distiller` (`train_and_quantize.py`) with `loss = α·CE + (1−α)·T²·KL(teacher/T ‖ student/T)`. The reference run reached **98.23%, +0.49 pp over the baseline student**: same size, better accuracy, from the teacher's dark knowledge.
4. **INT8 post-training quantization** writes `artifacts/kd_student_int8.tflite` at **18,816 B**, 3.2× smaller than FP32.
5. In MLflow, open experiment **tinymlops-kd-mnist-esp32** and the latest run. Show the params (`kd_alpha`, `kd_temperature`, `teacher_from_cache`), the metrics (`teacher_keras_test_accuracy`, `baseline_student_keras_test_accuracy`, `kd_student_keras_test_accuracy`, `kd_gain_vs_baseline_student`, `int8_tflite_bytes`) and the `.tflite` files under Artifacts. After Checkpoint 3 the same run also has the gate metrics and `deployment_gate=PASSED`.

**Connect it to Assignment 4.** Your exact Colab model goes through the same pipeline:

```bash
make import-colab     # python ml_pipeline/train_and_quantize.py --import-h5 models/colab/student_kd.h5 --import-tflite models/colab/student_kd_int8.tflite
```

It reproduces the Colab result exactly (**98.24%** Keras, then 98.30% INT8 after the gate). The import rebuilds the student from `config.yaml` and loads only the weights, so a model with the wrong architecture is rejected.

**✔ Verify:** `artifacts/metrics.json` exists, and MLflow shows teacher, baseline and KD student accuracies.

> Tip: start `make train` while you present Checkpoint 1.

---

## Checkpoint 3 — Automated quality gates and failure simulation

```bash
make gate             # or: python ml_pipeline/evaluate_and_gate.py
```

Expected output (your numbers will differ slightly):

```
[eval] 10,000 images in 6.8s | FP32=98.23% INT8=98.25% | FP32/INT8 top-1 agreement=99.93%
| GATE             | STATUS | MEASURED                                       | THRESHOLD                                |
| INT8_IO_QUANT    | PASS   | in int8 s=0.00392157 zp=-128 | out int8 ...    | int8 in/out, scale>0, zp in [-128,127]   |
| FLASH_SIZE       | PASS   | 18,816 B                                       | <= 35,000 B                              |
| OPERATOR_AUDIT   | PASS   | CONV_2D, MAX_POOL_2D, RESHAPE, FULLY_CONNECTED | subset of CONV_2D, MAX_POOL_2D, ...      |
| INT8_ACCURACY    | PASS   | 98.25%                                         | >= 96.00%                                |
| ACCURACY_DROP    | PASS   | -0.02 pp                                       | <= 1.50 pp                               |
[kd] teacher_keras_test_accuracy=99.33% | baseline_student_keras_test_accuracy=97.74% | kd_student_keras_test_accuracy=98.23% | kd_gain_vs_baseline_student=0.49pp
[GATE] ALL QUALITY GATES PASSED -> model approved for ESP32 deployment
```

Talking point: the KD gain is **reported, not gated**. On MNIST it's a fraction of a point and varies between runs (the Colab run gave +0.27 pp), so it mustn't block a deployment. The gates protect what the device needs: size, accuracy floor, quantization damage and supported operators.

**Failure simulation.** Tighten the flash ceiling to 10 KB:

```bash
make gate-fail-demo
# or: python ml_pipeline/evaluate_and_gate.py --max-bytes 10000 --report-out artifacts/gate_report_demo.json --golden-out artifacts/golden_vectors_demo.json --no-mlflow ; echo "exit=$?"
```

You'll see `FLASH_SIZE | FAIL`, then `[GATE] DEPLOYMENT BLOCKED -> failed gate(s): FLASH_SIZE`, then `exit=1`. In CI, that exit code stops the job before any firmware is built. The demo writes to separate `*_demo.json` files, so the approved model's report is left alone.

Other blocks you can show: `--min-accuracy 0.995` (accuracy floor) or `--max-drop 0.0` (quantization budget). Also show `test_operator_gate_allows_only_micro_ops`: a model with `SOFTMAX` or a custom op is rejected, because those kernels aren't registered in the firmware.

**✔ Verify:** the normal gate exits 0; the demo prints `DEPLOYMENT BLOCKED` and exits 1.

---

## Checkpoint 4 — C++ flatbuffer codegen and firmware compilation

```bash
make build-fw         # = make codegen + pio run -d firmware_esp32 + size report
# raw: python ml_pipeline/export_to_cpp.py && pio run -d firmware_esp32
```

What to show:
1. The top of `firmware_esp32/src/model_data.h`:
   - the model name and source (`pipeline_train` or `colab_import`)
   - the SHA-256
   - INT8 accuracy and the teacher / baseline / KD accuracies
   - the operator list and the input/output scale and zero-point
   - then `alignas(16) const unsigned char g_mnist_model_data[] = {...}`, since TFLite Micro needs 16-byte-aligned tensor data
2. `firmware_esp32/src/test_vectors.h`: `kTestImages[10][784]` (already INT8), `kTestLabels`, `kExpectedLogits` (the host reference outputs) and the quantization constants.
3. In `main.cpp`: `MicroMutableOpResolver<6>` (not `AllOpsResolver`) and `kTensorArenaSize = 32 * 1024`.
4. PlatformIO compiles for Xtensa LX6:

```
RAM:   [==        ]  18.2% (used 59680 bytes from 327680 bytes)
Flash: [===       ]  28.2% (used 369301 bytes from 1310720 bytes)

  ESP32-WROOM-32 budget
  Flash :    369301 B =   8.80% of 4 MB
  SRAM  :     59680 B =  11.21% of 520 KB (static DRAM incl. 32 KB tensor arena)
```

Talking point: the whole application, including the Arduino core, ESP-IDF, TensorFlow Lite Micro and the model, uses under 9% of the flash. The distilled model itself is **18.8 KB**, about 0.45% of the 4 MB flash.

**✔ Verify:** `pio run` ends with `SUCCESS`, and `firmware.bin` / `firmware.elf` exist in `firmware_esp32/.pio/build/esp32dev/`.

---

## Checkpoint 5 — Wokwi ESP32 simulation (BIST and interactive UART streaming)

### 5a. Headless simulation (wokwi-cli)

```bash
export WOKWI_CLI_TOKEN=wok_...         # PowerShell: $env:WOKWI_CLI_TOKEN="wok_..."
make sim
# raw: cd firmware_esp32 && wokwi-cli --timeout 120000 --scenario test_scenario.yaml --serial-log-file ../artifacts/sim_serial.log --fail-text "Guru Meditation" .
```

What the virtual ESP32 prints:

```
[TELEMETRY] MODEL_NAME=kd_student_int8 (pipeline_train)
[TELEMETRY] MODEL_SHA256_EMBEDDED=6b1f9a07...   MODEL_SHA256_RUNTIME =6b1f9a07...
[TELEMETRY] MODEL_INTEGRITY=OK
[TELEMETRY] ARENA_USED=... ARENA_SIZE=32768 ARENA_UTIL=..%
[BIST] Built-in self test: 10 golden MNIST digits (INT8)
| Digit | Expected | Predicted | Confidence | Latency (us) | Parity | Status |
|     0 |        0 |         0 |     99.xx% |      ...     | EXACT  | PASS   |
...
[TINYMLOPS_PARITY] BITEXACT=10/10
[TINYMLOPS_TEST_SUMMARY] PASSED=10/10 ACCURACY=100.00% AVG_LATENCY_US=... ARENA_BYTES=...
[STATUS] BIST PASSED - status LED (GPIO2) ON
```

The scenario then checks that the GPIO2 LED is high, that `PING` returns `PONG`, and that `SELFTEST:7` / `SELFTEST:3` return the correct prediction. It fails immediately on `Guru Meditation`, or by timeout on `[TINYMLOPS_FATAL]`. `artifacts/sim_report.md` is generated from the log.

Talking point on latency: Wokwi isn't cycle-accurate. Its simulated timer reported about 1 s per inference, so simulation only checks a loose 5 s ceiling, which still catches hangs. The real 100 ms budget is measured on a board with `make hil`.

**Failure demo (optional):** set `-DTINYMLOPS_TENSOR_ARENA_KB=4` in `platformio.ini` and run `make build-fw sim`. The firmware prints `[TINYMLOPS_FATAL] AllocateTensors() failed` and the scenario fails. Change it back afterwards.

### 5b. Interactive: watch it in VS Code and stream live MNIST frames

1. Install the **Wokwi Simulator** VS Code extension, open `firmware_esp32/`, press F1, and choose *Wokwi: Start Simulator*. The green LED blinks five times, then stays on.
2. `wokwi.toml` exposes the simulated UART on RFC2217 port 4000. In a second terminal:

```bash
make hil PORT=rfc2217://localhost:4000 EXTRA=50
# raw: python tests/serial_hil_verifier.py --port rfc2217://localhost:4000 --extra 50
```

The verifier first checks that the device's model SHA-256 matches the host's, so it refuses to run against stale firmware. It then sends each golden digit plus 50 random MNIST test images as `INFER:<784 int8>` and compares all 10 INT8 logits **bit for bit** with Python's `tf.lite.Interpreter`:

```
[hil] device model SHA-256 matches host (6b1f9a07...)
| golden[7]    |     7 |    7 |    7 |        ... | EXACT  |
[HIL_SUMMARY] SAMPLES=60 BITEXACT=60/60 DEVICE_ACCURACY=9x.xx% AVG_LATENCY_US=... BUDGET_US=100000
[HIL] PASS
```

The same command works with a physical board: run `make flash`, then `make hil PORT=COM5`. That's the only way to get the **real** inference latency.

Without the simulator or a board: `make host-parity` (`python tests/serial_hil_verifier.py --host-only`).

**✔ Verify:** `PASSED=10/10`, `BITEXACT=10/10`, `MODEL_INTEGRITY=OK`, the LED on, and `[HIL] PASS`.

---

## Checkpoint 6 — End-to-end CI/CD in GitHub Actions

```bash
# 1. Change a distillation hyperparameter (the teacher stays cached: its hash doesn't change)
sed -i 's/  temperature: 5 /  temperature: 3 /' ml_pipeline/config.yaml   # or edit it by hand
dvc params diff                                                          # shows exactly what changed
# 2. Push
git checkout -b demo/kd-t3 && git commit -am "exp: KD temperature 3" && git push -u origin demo/kd-t3
# 3. Open a PR to main, or: Actions → TinyMLOps KD Student MNIST ESP32 Pipeline → Run workflow
```

What to show in the Actions run:
1. **Job 1 · Distill, quantize, gate, codegen.** It restores the cached teacher, distills the student, then runs the gates, codegen, pytest and host parity. It uploads `tinyml-model` and `firmware-headers`.
2. **Job 2 · ESP32 build + Wokwi HIL simulation.** It builds the firmware with the new headers, boots it in Wokwi (`wokwi/wokwi-ci-action@v1`), and parses the serial log.
3. The **Summary** tab has three tables:
   * *Knowledge Distillation & Quality Gates*: teacher vs. baseline vs. KD student, FP32 vs. INT8, and the gate table
   * *ESP32-WROOM-32 Firmware Footprint*: flash as % of 4 MB and SRAM as % of 520 KB
   * *TinyMLOps Deployment & Simulation Report*: the BIST table, arena, parity and integrity
4. **Deploying the Colab model from CI:** run the workflow with **model_source = colab_import**.
5. **A blocked pipeline:** push `max_model_bytes: 10000`. Job 1 fails at the gate and Job 2 never runs.

> Don't press **Re-run jobs** on an older run while a new one is going. The workflow cancels in-progress runs on the same branch, so the older code's re-run would cancel the new one.

**✔ Verify:** both jobs are green, and the Step Summary has all three tables.

---

## Appendix — Q&A cheat sheet

| Likely question | Short answer |
|---|---|
| Why knowledge distillation? | The device can only afford a tiny model. KD lets a 35× smaller student learn from a strong teacher's soft predictions, not just the labels: here +0.3–0.5 pp over the same student trained alone, for free at inference time. |
| What do α and T do? | T = 5 softens the teacher's softmax so the student sees how similar the wrong digits are (the dark knowledge). The T² factor keeps the gradient scale comparable. α = 0.1 puts 90% of the weight on matching the teacher. |
| Why post-training quantization and not QAT? | It matches Assignment 4 and loses nothing here: INT8 is within ±0.1 pp of FP32 (98.23% → 98.25%). QAT would add training time with no measurable gain for this model. |
| Why `int8` input and output? | The MCU never touches floats on the inference path. The host quantizes images with the input scale (≈1/255) and zero-point (−128). |
| How do you know the device computes the same thing as Python? | TFLite Micro and `BUILTIN_REF` use identical integer kernels. The host logits are embedded in the firmware and compared bit for bit (`BITEXACT=10/10`), and the HIL verifier does the same for arbitrary images. |
| What stops a wrong model being flashed? | A SHA-256 chain: `metrics.json` → `golden_vectors.json` → `model_data.h` / `test_vectors.h` → recomputed on the device at boot (`MODEL_INTEGRITY`). The HIL verifier also refuses a device whose model hash differs from the host's. |
| Why a 32 KB arena? | The student's peak activations are about 6.8 KB (the first conv layer: 784 B in, 5,408 B out), plus TFLite Micro bookkeeping. Read the exact figure from `ARENA_USED` at boot. The rest is headroom, and a failed allocation shows up as `[TINYMLOPS_FATAL]`. |
| Why not deploy the teacher? | In INT8 it would technically fit (about 0.5 MB of flash), but it's 35× more parameters and far more multiply-accumulates per image: much slower and more energy per inference, for about 1 pp more accuracy. The student gets close at 18.8 KB, which leaves room for other firmware and models. |
