#include "model_runtime.h"

#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>

#include <math.h>
#include <new>
#include <string.h>

#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifndef TINYMLOPS_TENSOR_ARENA_KB
#define TINYMLOPS_TENSOR_ARENA_KB 32
#endif
#ifndef TFLITE_SCHEMA_VERSION
#define TFLITE_SCHEMA_VERSION 3
#endif

namespace tinymlops {
namespace {

constexpr int kTensorArenaSize = TINYMLOPS_TENSOR_ARENA_KB * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter micro_error_reporter;

// The six kernels the quality gate allows (ml_pipeline/config.yaml gates.allowed_ops).
tflite::MicroMutableOpResolver<6>& resolver() {
  static tflite::MicroMutableOpResolver<6> r;
  static bool init = false;
  if (!init) {
    r.AddConv2D();
    r.AddMaxPool2D();
    r.AddReshape();
    r.AddFullyConnected();
    r.AddQuantize();
    r.AddDequantize();
    init = true;
  }
  return r;
}

// Interpreter lives in static storage and is rebuilt with placement new on every model switch,
// so switching models never fragments the heap.
alignas(tflite::MicroInterpreter) uint8_t interpreter_storage[sizeof(tflite::MicroInterpreter)];
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

bool close_enough(float a, float b) { return fabsf(a - b) <= 1e-6f * fmaxf(1.f, fmaxf(fabsf(a), fabsf(b))); }

}  // namespace

void ModelRuntime::end() {
  if (interpreter) {
    interpreter->~MicroInterpreter();
    interpreter = nullptr;
  }
  input = output = nullptr;
  ready_ = false;
}

bool ModelRuntime::begin(const uint8_t* model_data, size_t len) {
  end();
  if (!model_data || len < 16 || ((uintptr_t)model_data & 15) != 0) {
    error_ = "model buffer missing or not 16-byte aligned";
    return false;
  }
  if (memcmp(model_data + 4, "TFL3", 4) != 0) {
    error_ = "not a TFLite flatbuffer";
    return false;
  }
  const tflite::Model* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_ = "unsupported TFLite schema version";
    return false;
  }
  interpreter = new (interpreter_storage)
      tflite::MicroInterpreter(model, resolver(), tensor_arena, kTensorArenaSize, &micro_error_reporter);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    end();
    error_ = "AllocateTensors() failed (arena too small or unsupported op)";
    return false;
  }
  input = interpreter->input(0);
  output = interpreter->output(0);
  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8 || output->dims->size < 2 ||
      output->dims->data[output->dims->size - 1] > kMaxClasses) {
    end();
    error_ = "unexpected tensor types/shapes (need int8 image -> int8 logits)";
    return false;
  }
  ready_ = true;
  error_ = "ok";
  return true;
}

size_t ModelRuntime::arena_used() const { return interpreter ? interpreter->arena_used_bytes() : 0; }
size_t ModelRuntime::arena_size() const { return kTensorArenaSize; }
size_t ModelRuntime::input_bytes() const { return input ? input->bytes : 0; }
int ModelRuntime::num_classes() const { return output ? output->dims->data[output->dims->size - 1] : 0; }
float ModelRuntime::input_scale() const { return input ? input->params.scale : 0.f; }
int ModelRuntime::input_zero_point() const { return input ? input->params.zero_point : 0; }
float ModelRuntime::output_scale() const { return output ? output->params.scale : 0.f; }
int ModelRuntime::output_zero_point() const { return output ? output->params.zero_point : 0; }

bool ModelRuntime::vectors_match(const VectorSet& v) const {
  return ready_ && v.image_size == input_bytes() && v.num_classes == num_classes() &&
         close_enough(v.input_scale, input_scale()) && v.input_zero_point == input_zero_point() &&
         close_enough(v.output_scale, output_scale()) && v.output_zero_point == output_zero_point();
}

void ModelRuntime::print_tensor_info() const {
  if (!ready_) return;
  Serial.printf("[TELEMETRY] ARENA_USED=%u ARENA_SIZE=%d ARENA_UTIL=%.1f%%\n", (unsigned)arena_used(),
                kTensorArenaSize, 100.0 * arena_used() / kTensorArenaSize);
  Serial.printf("[TELEMETRY] INPUT int8 [%d,%d,%d,%d] scale=%.8f zp=%d | OUTPUT int8 [%d,%d] scale=%.8f zp=%d\n",
                input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3],
                input->params.scale, (int)input->params.zero_point, output->dims->data[0], output->dims->data[1],
                output->params.scale, (int)output->params.zero_point);
}

InferenceResult ModelRuntime::infer(const int8_t* image) {
  InferenceResult r;
  if (!ready_) return r;
  memcpy(input->data.int8, image, input->bytes);
  const int64_t t0 = esp_timer_get_time();
  const TfLiteStatus st = interpreter->Invoke();
  r.latency_us = esp_timer_get_time() - t0;
  r.ok = (st == kTfLiteOk);
  if (!r.ok) return r;
  r.num_classes = num_classes();
  const float s = output->params.scale;
  const int zp = output->params.zero_point;
  int best = 0;
  for (int i = 0; i < r.num_classes; ++i) {
    r.logits[i] = output->data.int8[i];
    if (r.logits[i] > r.logits[best]) best = i;
  }
  const float top = (r.logits[best] - zp) * s;
  float denom = 0.f;
  for (int i = 0; i < r.num_classes; ++i) denom += expf((r.logits[i] - zp) * s - top);
  r.pred = best;
  r.score = r.logits[best];
  r.confidence = 100.f / denom;
  return r;
}

BistReport ModelRuntime::run_bist(const VectorSet& v, bool print_table) {
  BistReport rep;
  rep.total = v.count;
  if (!ready_ || !vectors_match(v)) {
    if (print_table) Serial.println("[BIST] vectors do not match this model - self test cannot run");
    rep.passed = 0;
    return rep;
  }
  if (print_table) {
    Serial.printf("[BIST] Built-in self test: %d golden MNIST digits (INT8)\n", v.count);
    Serial.println("+-------+----------+-----------+------------+--------------+--------+--------+");
    Serial.println("| Digit | Expected | Predicted | Confidence | Latency (us) | Parity | Status |");
    Serial.println("+-------+----------+-----------+------------+--------------+--------+--------+");
  }
  int64_t total_us = 0;
  rep.min_us = INT64_MAX;
  for (uint16_t i = 0; i < v.count; ++i) {
    InferenceResult r = infer(v.image(i));
    if (!r.ok) {
      Serial.printf("[TINYMLOPS_FATAL] Invoke() failed on vector %u\n", (unsigned)i);
      rep.passed = 0;
      return rep;
    }
    const bool ok = (r.pred == v.labels[i]);
    const bool exact = memcmp(r.logits, v.logits(i), v.num_classes) == 0;
    rep.passed += ok;
    rep.bitexact += exact;
    total_us += r.latency_us;
    if (r.latency_us < rep.min_us) rep.min_us = r.latency_us;
    if (r.latency_us > rep.max_us) rep.max_us = r.latency_us;
    if (print_table) {
      Serial.printf("| %5u | %8u | %9d | %9.2f%% | %12lld | %-6s | %-6s |\n", (unsigned)i, (unsigned)v.labels[i],
                    r.pred, r.confidence, (long long)r.latency_us, exact ? "EXACT" : "DIFF", ok ? "PASS" : "FAIL");
    }
  }
  rep.avg_us = (double)total_us / v.count;
  rep.arena_bytes = arena_used();
  if (print_table) {
    Serial.println("+-------+----------+-----------+------------+--------------+--------+--------+");
    Serial.printf("[TINYMLOPS_LATENCY] MIN_US=%lld MAX_US=%lld AVG_US=%.1f\n", (long long)rep.min_us,
                  (long long)rep.max_us, rep.avg_us);
    Serial.printf("[TINYMLOPS_PARITY] BITEXACT=%d/%d\n", rep.bitexact, rep.total);
    Serial.printf("[TINYMLOPS_TEST_SUMMARY] PASSED=%d/%d ACCURACY=%.2f%% AVG_LATENCY_US=%.1f ARENA_BYTES=%u\n",
                  rep.passed, rep.total, 100.0 * rep.passed / rep.total, rep.avg_us, (unsigned)rep.arena_bytes);
  }
  return rep;
}

}  // namespace tinymlops
