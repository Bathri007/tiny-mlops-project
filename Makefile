# =============================================================================
#  TinyMLOps MNIST -> ESP32-WROOM-32 : one-command shortcuts
#  Linux / macOS / Git-Bash on Windows (`choco install make` or `winget install GnuWin32.Make`)
# =============================================================================
SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

# Prefer the project virtualenv if it exists (Windows or POSIX layout)
ifneq ($(wildcard .venv/Scripts/python.exe),)
  PYTHON ?= .venv/Scripts/python.exe
  BIN := .venv/Scripts/
else ifneq ($(wildcard .venv/bin/python),)
  PYTHON ?= .venv/bin/python
  BIN := .venv/bin/
else
  PYTHON ?= python
  BIN :=
endif
PIO      ?= $(BIN)pio
WOKWI    ?= wokwi-cli
FW_DIR   := firmware_esp32
SIM_LOG  := artifacts/sim_serial.log
PORT     ?= rfc2217://localhost:4000
EXTRA    ?= 0

.PHONY: help setup train gate gate-fail-demo codegen build-fw fw-size sim sim-report hil host-parity \
        flash monitor test test-all mlflow-ui dvc-repro all clean

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-15s\033[0m %s\n",$$1,$$2}'

setup:  ## create .venv and install pinned deps
	python -m venv .venv
	$(MAKE) --no-print-directory _install

_install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

train:  ## Stage 1: train FP32 baseline + QAT INT8, log to MLflow
	$(PYTHON) ml_pipeline/train_and_quantize.py

gate:  ## Stage 2: quality gates (flash, accuracy, ops, quant) + golden vectors
	$(PYTHON) ml_pipeline/evaluate_and_gate.py

gate-fail-demo:  ## show the gate BLOCKING a model when the flash ceiling is 10 KB
	@echo ">>> Tightening Flash ceiling to 10,000 bytes - this deployment MUST be blocked"
	if $(PYTHON) ml_pipeline/evaluate_and_gate.py --max-bytes 10000 --report-out artifacts/gate_report_demo.json \
	     --golden-out artifacts/golden_vectors_demo.json --no-mlflow; then \
	  echo "!!! gate unexpectedly passed"; exit 1; \
	else \
	  echo ">>> Pipeline correctly blocked the oversized model (exit code 1)"; \
	fi

codegen:  ## Stage 3: .tflite + golden vectors -> model_data.h / test_vectors.h
	$(PYTHON) ml_pipeline/export_to_cpp.py

build-fw: codegen  ## compile ESP32 firmware (Xtensa LX6) and print Flash/RAM budget
	mkdir -p artifacts
	$(PIO) run -d $(FW_DIR) | tee artifacts/pio_build.log
	$(MAKE) --no-print-directory fw-size

fw-size:  ## Flash / RAM usage vs. the physical 4 MB / 520 KB of the ESP32-WROOM-32
	@awk '/^RAM:/  { gsub(/[()]/,""); for(i=1;i<=NF;i++) if($$i=="used") r=$$(i+1) } \
	     /^Flash:/{ gsub(/[()]/,""); for(i=1;i<=NF;i++) if($$i=="used") f=$$(i+1) } \
	     END { printf "\n  ESP32-WROOM-32 budget\n  Flash : %9d B = %6.2f%% of 4 MB\n  SRAM  : %9d B = %6.2f%% of 520 KB (static DRAM incl. 32 KB tensor arena)\n\n", \
	           f, 100*f/4194304, r, 100*r/532480 }' artifacts/pio_build.log

sim:  ## run firmware on the Wokwi ESP32 simulator (needs WOKWI_CLI_TOKEN)
	@test -n "$${WOKWI_CLI_TOKEN:-}" || { echo "Set WOKWI_CLI_TOKEN (https://wokwi.com/dashboard/ci)"; exit 2; }
	mkdir -p artifacts
	cd $(FW_DIR) && $(WOKWI) --timeout 120000 --scenario test_scenario.yaml \
	  --serial-log-file ../$(SIM_LOG) --fail-text "Guru Meditation" .
	cd .. && $(MAKE) --no-print-directory sim-report

sim-report:  ## parse the simulator serial log into a pass/fail report
	$(PYTHON) tests/serial_hil_verifier.py --parse-log $(SIM_LOG) --target sim --summary-md artifacts/sim_report.md

hil:  ## stream golden (+EXTRA) MNIST frames to a device: make hil PORT=COM5 EXTRA=100
	$(PYTHON) tests/serial_hil_verifier.py --port $(PORT) --extra $(EXTRA)

host-parity:  ## host TFLite reference == logits embedded in test_vectors.h
	$(PYTHON) tests/serial_hil_verifier.py --host-only

flash:  ## upload to a real ESP32 over USB
	$(PIO) run -d $(FW_DIR) -t upload

monitor:  ## open serial monitor on a real ESP32
	$(PIO) device monitor -d $(FW_DIR)

test:  ## Layer 1 pytest suite
	$(PYTHON) -m pytest -v tests/

test-all: train gate codegen test host-parity build-fw  ## full local pipeline (+ sim if token is set)
	if [ -n "$${WOKWI_CLI_TOKEN:-}" ]; then $(MAKE) --no-print-directory sim; \
	else echo ">>> WOKWI_CLI_TOKEN not set - skipping simulator stage"; fi

all: test-all

mlflow-ui:  ## open MLflow at http://127.0.0.1:5000
	$(BIN)mlflow ui --backend-store-uri ./mlruns --port 5000

dvc-repro:  ## reproduce the pipeline with DVC (only changed stages re-run)
	$(BIN)dvc repro

clean:  ## remove build + generated artifacts (keeps committed headers)
	rm -rf artifacts/*.tflite artifacts/*.json artifacts/*.log artifacts/*.md $(FW_DIR)/.pio .pytest_cache
