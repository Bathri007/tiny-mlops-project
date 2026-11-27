// =============================================================================
//  TinyMLOps MNIST inference firmware - ESP32-WROOM-32 + TensorFlow Lite Micro
// =============================================================================
//  Boot sequence
//    1. Telemetry  : firmware version, model SHA-256 (embedded vs. recomputed
//                    on-device), arena usage, free heap before/after
//                    AllocateTensors().
//    2. BIST       : runs the 10 golden INT8 MNIST digits from test_vectors.h,
//                    prints a table + machine-parseable summary line:
//       [TINYMLOPS_TEST_SUMMARY] PASSED=10/10 ACCURACY=100.00% AVG_LATENCY_US=.. ARENA_BYTES=..
//                    and blinks the GPIO2 status LED (then holds it ON) on 10/10.
//    3. Interactive UART mode (115200 8N1, newline-terminated commands):
//         INFER:<784 int8 values, comma/space separated>  or  INFER:<1568 hex chars>
//             -> {"pred":7,"score":118,"latency_us":1420,"confidence":99.71,"logits":[..]}
//         SELFTEST:<0-9>  run one embedded golden vector  -> JSON incl. label & parity
//         BIST            re-run the full built-in self test
//         INFO            re-print telemetry
//         PING            -> PONG
// =============================================================================
#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>

#include <math.h>
#include <string.h>

#include "esp_system.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "test_vectors.h"

#ifndef TINYMLOPS_FW_VERSION
#define TINYMLOPS_FW_VERSION "dev"
#endif
#ifndef TINYMLOPS_TENSOR_ARENA_KB
#define TINYMLOPS_TENSOR_ARENA_KB 32
#endif
#ifndef TINYMLOPS_LED_PIN
#define TINYMLOPS_LED_PIN 2
#endif
#ifndef TFLITE_SCHEMA_VERSION
#define TFLITE_SCHEMA_VERSION 3
#endif

namespace {

// ---------------------------------------------------------------------------
// TFLite Micro globals
// ---------------------------------------------------------------------------
constexpr int kTensorArenaSize = TINYMLOPS_TENSOR_ARENA_KB * 1024;  // 32 KB
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

bool g_model_ready = false;
bool g_integrity_ok = false;
uint32_t g_heap_before_alloc = 0;
uint32_t g_heap_after_alloc = 0;
char g_runtime_sha256[65] = {0};

// ---------------------------------------------------------------------------
// Minimal SHA-256 (FIPS 180-4) - verifies the flatbuffer in flash matches the
// hash stamped by the CI pipeline, without depending on HW crypto peripherals.
// ---------------------------------------------------------------------------
constexpr uint32_t kSha256K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

void sha256_block(uint32_t h[8], const uint8_t blk[64]) {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (uint32_t)blk[4 * i] << 24 | (uint32_t)blk[4 * i + 1] << 16 | (uint32_t)blk[4 * i + 2] << 8 |
           (uint32_t)blk[4 * i + 3];
  }
  for (int i = 16; i < 64; ++i) {
    uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
  for (int i = 0; i < 64; ++i) {
    uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    uint32_t ch = (e & f) ^ (~e & g);
    uint32_t t1 = hh + S1 + ch + kSha256K[i] + w[i];
    uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t t2 = S0 + maj;
    hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

void sha256_hex(const uint8_t* data, size_t len, char out_hex[65]) {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  size_t full = len / 64;
  for (size_t i = 0; i < full; ++i) sha256_block(h, data + 64 * i);
  uint8_t tail[128] = {0};
  size_t rem = len - full * 64;
  memcpy(tail, data + full * 64, rem);
  tail[rem] = 0x80;
  size_t tail_len = (rem < 56) ? 64 : 128;
  uint64_t bits = (uint64_t)len * 8;
  for (int i = 0; i < 8; ++i) tail[tail_len - 1 - i] = (uint8_t)(bits >> (8 * i));
  sha256_block(h, tail);
  if (tail_len == 128) sha256_block(h, tail + 64);
  for (int i = 0; i < 8; ++i) snprintf(out_hex + 8 * i, 9, "%08x", (unsigned)h[i]);
  out_hex[64] = '\0';
}

// ---------------------------------------------------------------------------
// Inference helpers
// ---------------------------------------------------------------------------
struct InferenceResult {
  int pred;
  int8_t score;        // raw int8 logit of the winning class
  float confidence;    // softmax(dequantized logits)[pred] in %
  int64_t latency_us;  // Invoke() wall time
  int8_t logits[kNumClasses];
  bool ok;
};

InferenceResult run_inference(const int8_t* image) {
  InferenceResult r{};
  memcpy(input->data.int8, image, kImageSize);
  const int64_t t0 = esp_timer_get_time();
  TfLiteStatus st = interpreter->Invoke();
  r.latency_us = esp_timer_get_time() - t0;
  r.ok = (st == kTfLiteOk);
  if (!r.ok) return r;

  const float out_scale = output->params.scale;
  const int out_zp = output->params.zero_point;
  int best = 0;
  for (int i = 0; i < kNumClasses; ++i) {
    r.logits[i] = output->data.int8[i];
    if (r.logits[i] > r.logits[best]) best = i;
  }
  float max_logit = (r.logits[best] - out_zp) * out_scale;
  float denom = 0.f;
  for (int i = 0; i < kNumClasses; ++i) denom += expf((r.logits[i] - out_zp) * out_scale - max_logit);
  r.pred = best;
  r.score = r.logits[best];
  r.confidence = 100.f / denom;
  return r;
}

void print_logits_json(const int8_t* logits) {
  Serial.print("[");
  for (int i = 0; i < kNumClasses; ++i) {
    Serial.print(logits[i]);
    if (i + 1 < kNumClasses) Serial.print(",");
  }
  Serial.print("]");
}

void set_led(bool on) { digitalWrite(TINYMLOPS_LED_PIN, on ? HIGH : LOW); }

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------
void print_telemetry() {
  Serial.println();
  Serial.println("==================================================================");
  Serial.println(" TinyMLOps | MNIST CNN on ESP32-WROOM-32 | TensorFlow Lite Micro");
  Serial.println("==================================================================");
  Serial.printf("[TELEMETRY] FW_VERSION=%s BUILD=%s %s\n", TINYMLOPS_FW_VERSION, __DATE__, __TIME__);
  Serial.printf("[TELEMETRY] CHIP=%s REV=%d CORES=%d CPU_MHZ=%u FLASH_BYTES=%u\n", ESP.getChipModel(),
                ESP.getChipRevision(), ESP.getChipCores(), (unsigned)ESP.getCpuFreqMHz(),
                (unsigned)ESP.getFlashChipSize());
  Serial.printf("[TELEMETRY] MODEL_BYTES=%u MODEL_GENERATED=%s\n", g_mnist_model_data_len, kModelGeneratedAt);
  Serial.printf("[TELEMETRY] MODEL_SHA256_EMBEDDED=%s\n", kModelSha256);
  Serial.printf("[TELEMETRY] MODEL_SHA256_RUNTIME =%s\n", g_runtime_sha256);
  Serial.printf("[TELEMETRY] MODEL_INTEGRITY=%s\n", g_integrity_ok ? "OK" : "MISMATCH");
  Serial.printf("[TELEMETRY] HEAP_TOTAL=%u HEAP_FREE_BEFORE_ALLOC=%u HEAP_FREE_AFTER_ALLOC=%u\n",
                (unsigned)ESP.getHeapSize(), (unsigned)g_heap_before_alloc, (unsigned)g_heap_after_alloc);
  if (g_model_ready) {
    const size_t used = interpreter->arena_used_bytes();
    Serial.printf("[TELEMETRY] ARENA_USED=%u ARENA_SIZE=%d ARENA_UTIL=%.1f%%\n", (unsigned)used,
                  kTensorArenaSize, 100.0 * used / kTensorArenaSize);
    Serial.printf("[TELEMETRY] INPUT int8 [%d,%d,%d,%d] scale=%.8f zp=%d | OUTPUT int8 [%d,%d] scale=%.8f zp=%d\n",
                  input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3],
                  input->params.scale, (int)input->params.zero_point, output->dims->data[0],
                  output->dims->data[1], output->params.scale, (int)output->params.zero_point);
  }
  Serial.println("------------------------------------------------------------------");
}

// ---------------------------------------------------------------------------
// Model bring-up
// ---------------------------------------------------------------------------
bool setup_model() {
  sha256_hex(g_mnist_model_data, g_mnist_model_data_len, g_runtime_sha256);
  g_integrity_ok = (strcmp(g_runtime_sha256, kModelSha256) == 0) &&
                   (strcmp(kModelSha256, TINYMLOPS_VECTORS_MODEL_SHA256) == 0);

  model = tflite::GetModel(g_mnist_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[TINYMLOPS_FATAL] Model schema v%d != supported v%d\n", (int)model->version(),
                  TFLITE_SCHEMA_VERSION);
    return false;
  }

  // Only the 6 kernels the quality gate allows -> minimal flash footprint.
  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize,
                                                     error_reporter);
  interpreter = &static_interpreter;

  g_heap_before_alloc = ESP.getFreeHeap();
  TfLiteStatus st = interpreter->AllocateTensors();
  g_heap_after_alloc = ESP.getFreeHeap();
  if (st != kTfLiteOk) {
    Serial.printf("[TINYMLOPS_FATAL] AllocateTensors() failed (arena=%d bytes)\n", kTensorArenaSize);
    return false;
  }
  input = interpreter->input(0);
  output = interpreter->output(0);
  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8 || input->bytes != (size_t)kImageSize) {
    Serial.println("[TINYMLOPS_FATAL] Unexpected tensor type/shape (need int8 [1,28,28,1] -> int8 [1,10])");
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Built-in self test
// ---------------------------------------------------------------------------
bool run_bist() {
  Serial.println("[BIST] Built-in self test: 10 golden MNIST digits (INT8)");
  Serial.println("+-------+----------+-----------+------------+--------------+--------+--------+");
  Serial.println("| Digit | Expected | Predicted | Confidence | Latency (us) | Parity | Status |");
  Serial.println("+-------+----------+-----------+------------+--------------+--------+--------+");
  int passed = 0, bitexact = 0;
  int64_t total_us = 0, min_us = INT64_MAX, max_us = 0;
  for (int i = 0; i < kNumTestVectors; ++i) {
    InferenceResult r = run_inference(kTestImages[i]);
    if (!r.ok) {
      Serial.printf("[TINYMLOPS_FATAL] Invoke() failed on vector %d\n", i);
      return false;
    }
    const bool ok = (r.pred == kTestLabels[i]);
    const bool exact = memcmp(r.logits, kExpectedLogits[i], kNumClasses) == 0;
    passed += ok;
    bitexact += exact;
    total_us += r.latency_us;
    if (r.latency_us < min_us) min_us = r.latency_us;
    if (r.latency_us > max_us) max_us = r.latency_us;
    Serial.printf("| %5d | %8d | %9d | %9.2f%% | %12lld | %-6s | %-6s |\n", i, kTestLabels[i], r.pred,
                  r.confidence, (long long)r.latency_us, exact ? "EXACT" : "DIFF", ok ? "PASS" : "FAIL");
  }
  Serial.println("+-------+----------+-----------+------------+--------------+--------+--------+");
  const double avg = (double)total_us / kNumTestVectors;
  const size_t arena = interpreter->arena_used_bytes();
  Serial.printf("[TINYMLOPS_LATENCY] MIN_US=%lld MAX_US=%lld AVG_US=%.1f\n", (long long)min_us,
                (long long)max_us, avg);
  Serial.printf("[TINYMLOPS_PARITY] BITEXACT=%d/%d\n", bitexact, kNumTestVectors);
  Serial.printf("[TINYMLOPS_TEST_SUMMARY] PASSED=%d/%d ACCURACY=%.2f%% AVG_LATENCY_US=%.1f ARENA_BYTES=%u\n",
                passed, kNumTestVectors, 100.0 * passed / kNumTestVectors, avg, (unsigned)arena);
  return passed == kNumTestVectors;
}

void signal_result(bool pass) {
  if (pass) {
    for (int i = 0; i < 5; ++i) {  // 5 blinks, then hold ON = "deployment healthy"
      set_led(true);
      delay(150);
      set_led(false);
      delay(150);
    }
    set_led(true);
    Serial.println("[STATUS] BIST PASSED - status LED (GPIO2) ON");
  } else {
    set_led(false);
    Serial.println("[STATUS] BIST FAILED - status LED (GPIO2) OFF");
  }
}

// ---------------------------------------------------------------------------
// UART command interface
// ---------------------------------------------------------------------------
constexpr size_t kLineMax = 4096;  // 784 * "-128," = 3920 chars + prefix
char g_line[kLineMax];
size_t g_line_len = 0;
bool g_line_overflow = false;
int8_t g_frame[kImageSize];

int hex_nibble(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// Parses either 784 decimal int8 values (comma / whitespace separated) or a
// 1568-char hex string (two's-complement bytes). Returns number of values parsed
// or -1 on a syntax/range error.
int parse_frame(const char* s, int8_t* out) {
  while (*s == ' ') ++s;
  const size_t n = strlen(s);
  bool is_hex = (n == 2 * (size_t)kImageSize);
  for (size_t i = 0; is_hex && i < n; ++i) is_hex = hex_nibble(s[i]) >= 0;
  if (is_hex) {
    for (int i = 0; i < kImageSize; ++i) out[i] = (int8_t)((hex_nibble(s[2 * i]) << 4) | hex_nibble(s[2 * i + 1]));
    return kImageSize;
  }
  int count = 0;
  const char* p = s;
  while (*p) {
    while (*p == ',' || *p == ' ' || *p == '\t') ++p;
    if (!*p) break;
    char* end = nullptr;
    long v = strtol(p, &end, 10);
    if (end == p || v < -128 || v > 127) return -1;
    if (count >= kImageSize) return count + 1;  // too many values
    out[count++] = (int8_t)v;
    p = end;
  }
  return count;
}

void respond_inference(const InferenceResult& r, int label, int index) {
  Serial.printf("{");
  if (index >= 0) Serial.printf("\"idx\":%d,\"label\":%d,", index, label);
  Serial.printf("\"pred\":%d,\"score\":%d,\"latency_us\":%lld,\"confidence\":%.2f,\"logits\":", r.pred,
                (int)r.score, (long long)r.latency_us, r.confidence);
  print_logits_json(r.logits);
  if (index >= 0) {
    Serial.printf(",\"parity\":\"%s\"", memcmp(r.logits, kExpectedLogits[index], kNumClasses) == 0 ? "EXACT" : "DIFF");
  }
  Serial.println("}");
}

void handle_line(char* line) {
  // strip trailing CR / spaces
  size_t n = strlen(line);
  while (n && (line[n - 1] == '\r' || line[n - 1] == ' ')) line[--n] = '\0';
  if (n == 0) return;

  if (!g_model_ready) {
    Serial.println("{\"error\":\"model not initialised\"}");
    return;
  }
  if (strncmp(line, "INFER:", 6) == 0) {
    int count = parse_frame(line + 6, g_frame);
    if (count != kImageSize) {
      Serial.printf("{\"error\":\"expected %d int8 values, got %d\"}\n", kImageSize, count);
      return;
    }
    InferenceResult r = run_inference(g_frame);
    if (!r.ok) {
      Serial.println("{\"error\":\"Invoke failed\"}");
      return;
    }
    respond_inference(r, -1, -1);
  } else if (strncmp(line, "SELFTEST:", 9) == 0) {
    int idx = atoi(line + 9);
    if (idx < 0 || idx >= kNumTestVectors) {
      Serial.println("{\"error\":\"index must be 0..9\"}");
      return;
    }
    InferenceResult r = run_inference(kTestImages[idx]);
    respond_inference(r, kTestLabels[idx], idx);
  } else if (strcmp(line, "BIST") == 0) {
    signal_result(run_bist());
  } else if (strcmp(line, "INFO") == 0) {
    print_telemetry();
  } else if (strcmp(line, "PING") == 0) {
    Serial.println("PONG");
  } else {
    Serial.println("{\"error\":\"unknown command (INFER:, SELFTEST:<n>, BIST, INFO, PING)\"}");
  }
}

void poll_serial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      g_line[g_line_len] = '\0';
      if (g_line_overflow) {
        Serial.printf("{\"error\":\"line longer than %u bytes\"}\n", (unsigned)(kLineMax - 1));
      } else {
        handle_line(g_line);
      }
      g_line_len = 0;
      g_line_overflow = false;
    } else if (g_line_len < kLineMax - 1) {
      g_line[g_line_len++] = c;
    } else {
      g_line_overflow = true;
    }
  }
}

}  // namespace

// =============================================================================
void setup() {
  pinMode(TINYMLOPS_LED_PIN, OUTPUT);
  set_led(false);
  Serial.setRxBufferSize(kLineMax);
  Serial.begin(115200);
  delay(200);

  g_model_ready = setup_model();
  print_telemetry();
  if (!g_model_ready) {
    signal_result(false);
    return;
  }
  if (!g_integrity_ok) {
    Serial.println("[WARN] Model SHA-256 mismatch between flash, model_data.h and test_vectors.h");
  }
  signal_result(run_bist());
  Serial.println("[TINYMLOPS_READY] Commands: INFER:<784 int8>, SELFTEST:<0-9>, BIST, INFO, PING");
}

void loop() {
  poll_serial();
  delay(1);
}
