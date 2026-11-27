// =============================================================================
//  TinyMLOps inference firmware - knowledge-distilled MNIST student CNN
//  (14,410 params, full-INT8) on ESP32-WROOM-32 + TensorFlow Lite Micro,
//  with over-the-air (WiFi) MODEL updates from GitHub Releases.
// =============================================================================
//  Boot
//    1. Mount LittleFS, read /state.txt, run the active model (slot A/B) or the
//       factory model compiled into this firmware (model_data.h). A slot model whose
//       SHA-256 or self-test fails falls back to factory automatically.
//    2. Telemetry + BIST on the active model's golden vectors, machine-parseable:
//       [TINYMLOPS_TEST_SUMMARY] PASSED=10/10 ACCURACY=100.00% AVG_LATENCY_US=.. ARENA_BYTES=..
//       GPIO2 LED: 5 blinks then steady ON when the self-test passes.
//    3. OTA task (core 0): WiFi -> manifest -> download -> verify -> self-test -> switch
//       or roll back (see ota_agent.h). Serial keeps working meanwhile (core 1).
//  Serial commands (115200 8N1, newline-terminated)
//    INFER:<784 int8 comma/space separated> | INFER:<1568 hex>  -> {"pred":..,"logits":[..]}
//    SELFTEST:<i>  one golden vector of the active model       -> JSON incl. label & parity
//    BIST          full self test        INFO   telemetry      PING -> PONG
//    OTA           check for a new model now   STATUS  OTA + slot status
//    FACTORY       switch back to the factory model (persists across reboots)
// =============================================================================
#include <Arduino.h>

#include <string.h>

#include "deployment.h"
#include "model_data.h"
#include "model_runtime.h"
#include "model_store.h"
#include "ota_agent.h"
#include "test_vectors.h"

#if __has_include("secrets.h")
#include "secrets.h"
#endif

#ifndef TINYMLOPS_FW_VERSION
#define TINYMLOPS_FW_VERSION "dev"
#endif
#ifndef TINYMLOPS_LED_PIN
#define TINYMLOPS_LED_PIN 2
#endif
#ifndef TINYMLOPS_BOOT_BUTTON_PIN
#define TINYMLOPS_BOOT_BUTTON_PIN 0
#endif
#ifndef TINYMLOPS_OTA_ENABLED
#define TINYMLOPS_OTA_ENABLED 1
#endif
#ifndef TINYMLOPS_OTA_INTERVAL_S
#define TINYMLOPS_OTA_INTERVAL_S 300
#endif
#ifndef TINYMLOPS_OTA_TLS_INSECURE
#define TINYMLOPS_OTA_TLS_INSECURE 0
#endif
#ifndef TINYMLOPS_OTA_REQUIRE_BITEXACT
#define TINYMLOPS_OTA_REQUIRE_BITEXACT 1
#endif
#ifndef TINYMLOPS_WIFI_SSID
#define TINYMLOPS_WIFI_SSID "Wokwi-GUEST"  // Wokwi simulator's open network
#define TINYMLOPS_WIFI_PASSWORD ""
#endif
#ifndef TINYMLOPS_MANIFEST_URL
#define TINYMLOPS_MANIFEST_URL \
  "https://github.com/Bathri007/tiny-mlops-project/releases/latest/download/manifest.json"
#endif

using namespace tinymlops;

namespace {

constexpr char kSupportedOps[] = "CONV_2D,MAX_POOL_2D,RESHAPE,FULLY_CONNECTED,QUANTIZE,DEQUANTIZE";

ModelRuntime g_runtime;
ModelStore g_store;
Deployment g_deployment(g_runtime, g_store);
uint32_t g_heap_before = 0, g_heap_after = 0;

void set_led(bool on) { digitalWrite(TINYMLOPS_LED_PIN, on ? HIGH : LOW); }

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

void print_header() {
  Serial.println();
  Serial.println("==================================================================");
  Serial.println(" TinyMLOps | KD Student MNIST CNN on ESP32-WROOM-32 | TFLite Micro");
  Serial.println("==================================================================");
  Serial.printf("[TELEMETRY] FW_VERSION=%s BUILD=%s %s\n", TINYMLOPS_FW_VERSION, __DATE__, __TIME__);
  Serial.printf("[TELEMETRY] CHIP=%s REV=%d CORES=%d CPU_MHZ=%u FLASH_BYTES=%u\n", ESP.getChipModel(),
                ESP.getChipRevision(), ESP.getChipCores(), (unsigned)ESP.getCpuFreqMHz(),
                (unsigned)ESP.getFlashChipSize());
  Serial.printf("[TELEMETRY] FACTORY_MODEL_BYTES=%u FACTORY_MODEL_GENERATED=%s\n", g_mnist_model_data_len,
                kModelGeneratedAt);
}

void print_info() {
  print_header();
  Serial.printf("[TELEMETRY] HEAP_TOTAL=%u HEAP_FREE_BEFORE_ALLOC=%u HEAP_FREE_AFTER_ALLOC=%u\n",
                (unsigned)ESP.getHeapSize(), (unsigned)g_heap_before, (unsigned)g_heap_after);
  g_deployment.print_telemetry();
  Serial.println("------------------------------------------------------------------");
}

// ---------------------------------------------------------------------------
// UART command interface
// ---------------------------------------------------------------------------
constexpr size_t kLineMax = 4096;  // 784 * "-128," = 3920 chars + prefix
constexpr size_t kMaxImage = 4096;
char g_line[kLineMax];
size_t g_line_len = 0;
bool g_line_overflow = false;
int8_t g_frame[kMaxImage];

int hex_nibble(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// Parses `want` int8 values (comma/space separated decimals) or 2*want hex chars.
int parse_frame(const char* s, int8_t* out, int want) {
  while (*s == ' ') ++s;
  const size_t n = strlen(s);
  bool is_hex = (n == 2 * (size_t)want);
  for (size_t i = 0; is_hex && i < n; ++i) is_hex = hex_nibble(s[i]) >= 0;
  if (is_hex) {
    for (int i = 0; i < want; ++i) out[i] = (int8_t)((hex_nibble(s[2 * i]) << 4) | hex_nibble(s[2 * i + 1]));
    return want;
  }
  int count = 0;
  const char* p = s;
  while (*p) {
    while (*p == ',' || *p == ' ' || *p == '\t') ++p;
    if (!*p) break;
    char* end = nullptr;
    long v = strtol(p, &end, 10);
    if (end == p || v < -128 || v > 127) return -1;
    if (count >= want) return count + 1;
    out[count++] = (int8_t)v;
    p = end;
  }
  return count;
}

// Built into one buffer and written with a single call: the OTA task logs from the other core,
// and a reply split across several writes could get an [OTA] line spliced into it.
void respond(const InferenceResult& r, int label, int index, bool exact) {
  char buf[320];
  int n = 0;
  n += snprintf(buf + n, sizeof(buf) - n, "{");
  if (index >= 0) n += snprintf(buf + n, sizeof(buf) - n, "\"idx\":%d,\"label\":%d,", index, label);
  n += snprintf(buf + n, sizeof(buf) - n, "\"pred\":%d,\"score\":%d,\"latency_us\":%lld,\"confidence\":%.2f,\"logits\":[",
                r.pred, (int)r.score, (long long)r.latency_us, r.confidence);
  for (int i = 0; i < r.num_classes; ++i) n += snprintf(buf + n, sizeof(buf) - n, i ? ",%d" : "%d", r.logits[i]);
  n += snprintf(buf + n, sizeof(buf) - n, "]");
  if (index >= 0) n += snprintf(buf + n, sizeof(buf) - n, ",\"parity\":\"%s\"", exact ? "EXACT" : "DIFF");
  snprintf(buf + n, sizeof(buf) - n, "}\n");
  Serial.print(buf);
}

void handle_line(char* line) {
  size_t n = strlen(line);
  while (n && (line[n - 1] == '\r' || line[n - 1] == ' ')) line[--n] = '\0';
  if (n == 0) return;

  if (strncmp(line, "INFER:", 6) == 0) {
    const int want = (int)g_deployment.image_size();
    const int count = parse_frame(line + 6, g_frame, want);
    if (count != want) {
      Serial.printf("{\"error\":\"expected %d int8 values, got %d\"}\n", want, count);
      return;
    }
    InferenceResult r = g_deployment.infer(g_frame);
    if (!r.ok) {
      Serial.println("{\"error\":\"Invoke failed\"}");
      return;
    }
    respond(r, -1, -1, false);
  } else if (strncmp(line, "SELFTEST:", 9) == 0) {
    const int idx = atoi(line + 9);
    InferenceResult r;
    uint8_t label = 0;
    bool exact = false;
    if (idx < 0 || !g_deployment.selftest((uint16_t)idx, &r, &label, &exact)) {
      Serial.printf("{\"error\":\"index must be 0..%d\"}\n", g_deployment.vector_count() - 1);
      return;
    }
    respond(r, label, idx, exact);
  } else if (strcmp(line, "BIST") == 0) {
    signal_result(g_deployment.run_bist(true).all_passed());
  } else if (strcmp(line, "INFO") == 0) {
    print_info();
  } else if (strcmp(line, "PING") == 0) {
    Serial.println("PONG");
  } else if (strcmp(line, "OTA") == 0) {
#if TINYMLOPS_OTA_ENABLED
    Serial.println("[OTA] manual check requested");
    ota_trigger();
#else
    Serial.println("{\"error\":\"OTA disabled in this build\"}");
#endif
  } else if (strcmp(line, "STATUS") == 0) {
    const ActiveState s = g_deployment.state();
    Serial.printf("[STATUS] MODEL=%s SLOT=%c VERSION=%ld SHA256=%.16s...\n", g_deployment.model_name(), s.slot,
                  (long)s.version, s.sha);
#if TINYMLOPS_OTA_ENABLED
    ota_print_status();
#endif
  } else if (strcmp(line, "FACTORY") == 0) {
    const bool ok = g_deployment.force_factory();
    Serial.printf("[STATUS] switched to factory model: %s\n", ok ? "OK" : "FAILED");
    signal_result(g_deployment.run_bist(true).all_passed());
  } else {
    Serial.println("{\"error\":\"unknown command (INFER:, SELFTEST:<n>, BIST, INFO, PING, OTA, STATUS, FACTORY)\"}");
  }
}

void poll_serial() {
  while (Serial.available() > 0) {
    const char c = (char)Serial.read();
    if (c == '\n') {
      g_line[g_line_len] = '\0';
      if (g_line_overflow) Serial.printf("{\"error\":\"line longer than %u bytes\"}\n", (unsigned)(kLineMax - 1));
      else handle_line(g_line);
      g_line_len = 0;
      g_line_overflow = false;
    } else if (g_line_len < kLineMax - 1) {
      g_line[g_line_len++] = c;
    } else {
      g_line_overflow = true;
    }
  }
}

void poll_boot_button() {
#if TINYMLOPS_OTA_ENABLED
  static uint32_t pressed_at = 0;
  const bool down = digitalRead(TINYMLOPS_BOOT_BUTTON_PIN) == LOW;
  if (down && pressed_at == 0) pressed_at = millis();
  if (!down && pressed_at && millis() - pressed_at > 50) {
    Serial.println("[OTA] BOOT button -> check for a new model");
    ota_trigger();
  }
  if (!down) pressed_at = 0;
#endif
}

}  // namespace

// =============================================================================
void setup() {
  pinMode(TINYMLOPS_LED_PIN, OUTPUT);
  pinMode(TINYMLOPS_BOOT_BUTTON_PIN, INPUT_PULLUP);
  set_led(false);
  Serial.setRxBufferSize(kLineMax);
  Serial.begin(115200);
  delay(200);

  print_header();
  if (strcmp(kModelSha256, TINYMLOPS_VECTORS_MODEL_SHA256) != 0) {
    Serial.println("[WARN] model_data.h and test_vectors.h come from different models");
  }
  if (!g_store.begin()) Serial.println("[WARN] LittleFS mount failed - only the factory model is available");

  VectorSet factory;
  factory.count = kNumTestVectors;
  factory.image_size = kImageSize;
  factory.num_classes = kNumClasses;
  factory.input_scale = kInputScale;
  factory.input_zero_point = kInputZeroPoint;
  factory.output_scale = kOutputScale;
  factory.output_zero_point = kOutputZeroPoint;
  factory.labels = kTestLabels;
  factory.expected_logits = &kExpectedLogits[0][0];
  factory.images = &kTestImages[0][0];
  g_deployment.set_factory(g_mnist_model_data, g_mnist_model_data_len, kModelSha256, kModelName, factory);
  g_deployment.set_require_bitexact(TINYMLOPS_OTA_REQUIRE_BITEXACT);

  g_heap_before = ESP.getFreeHeap();
  const bool pass = g_deployment.boot();
  g_heap_after = ESP.getFreeHeap();
  Serial.printf("[TELEMETRY] HEAP_TOTAL=%u HEAP_FREE_BEFORE_ALLOC=%u HEAP_FREE_AFTER_ALLOC=%u\n",
                (unsigned)ESP.getHeapSize(), (unsigned)g_heap_before, (unsigned)g_heap_after);
  signal_result(pass);
  Serial.println("[TINYMLOPS_READY] Commands: INFER:<784 int8>, SELFTEST:<0-9>, BIST, INFO, PING, OTA, STATUS, FACTORY");

#if TINYMLOPS_OTA_ENABLED
  OtaConfig cfg;
  cfg.wifi_ssid = TINYMLOPS_WIFI_SSID;
  cfg.wifi_password = TINYMLOPS_WIFI_PASSWORD;
  cfg.manifest_url = TINYMLOPS_MANIFEST_URL;
  cfg.firmware_version = TINYMLOPS_FW_VERSION;
  cfg.supported_ops = kSupportedOps;
  cfg.interval_s = TINYMLOPS_OTA_INTERVAL_S;
  cfg.tls_insecure = TINYMLOPS_OTA_TLS_INSECURE;
  cfg.led_pin = TINYMLOPS_LED_PIN;
  ota_start(&g_deployment, &g_store, cfg);
#endif
}

void loop() {
  poll_serial();
  poll_boot_button();
  delay(1);
}
