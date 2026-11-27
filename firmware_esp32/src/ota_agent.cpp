#include "ota_agent.h"

#include <Arduino.h>
#include <FS.h>
#include <HTTPClient.h>
#include <LittleFS.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>

#include <time.h>

#include "ca_roots.h"
#include "manifest.h"
#include "sha256.h"

namespace tinymlops {
namespace {

Deployment* g_dep = nullptr;
ModelStore* g_store = nullptr;
OtaConfig g_cfg;
TaskHandle_t g_task = nullptr;
char g_last_status[24] = "NOT_RUN";
char g_last_detail[64] = "";
int32_t g_checks = 0;

constexpr size_t kManifestMax = 4096;
constexpr uint32_t kHttpTimeoutMs = 20000;

void result(const char* status, const char* detail, int32_t version) {
  strlcpy(g_last_status, status, sizeof(g_last_status));
  strlcpy(g_last_detail, detail, sizeof(g_last_detail));
  const ActiveState s = g_dep->state();
  Serial.printf("[TINYMLOPS_OTA_RESULT] STATUS=%s MANIFEST_VERSION=%ld ACTIVE_VERSION=%ld ACTIVE_SLOT=%c REASON=%s\n",
                status, (long)version, (long)s.version, s.slot, detail);
}

void blink(int times, int on_ms, int off_ms) {
  if (g_cfg.led_pin < 0) return;
  for (int i = 0; i < times; ++i) {
    digitalWrite(g_cfg.led_pin, HIGH);
    delay(on_ms);
    digitalWrite(g_cfg.led_pin, LOW);
    delay(off_ms);
  }
  digitalWrite(g_cfg.led_pin, HIGH);  // steady ON = a self-tested model is running
}

bool wifi_up() {
  if (WiFi.status() == WL_CONNECTED) return true;
  Serial.printf("[OTA] WIFI connecting to \"%s\" ...\n", g_cfg.wifi_ssid);
  WiFi.mode(WIFI_STA);
  // Wokwi's virtual access point lives on channel 6; passing it skips the channel scan there.
  if (strcmp(g_cfg.wifi_ssid, "Wokwi-GUEST") == 0) WiFi.begin(g_cfg.wifi_ssid, g_cfg.wifi_password, 6);
  else WiFi.begin(g_cfg.wifi_ssid, g_cfg.wifi_password);
  const uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 30000) delay(250);
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[OTA] WIFI FAILED (check secrets.h SSID/password; the ESP32 needs 2.4 GHz)");
    return false;
  }
  Serial.printf("[OTA] WIFI_CONNECTED IP=%s RSSI=%d\n", WiFi.localIP().toString().c_str(), WiFi.RSSI());
  return true;
}

// TLS certificate validation needs the real date.
bool time_synced() {
  if (time(nullptr) > 1700000000) return true;
  configTime(0, 0, "pool.ntp.org", "time.google.com");
  const uint32_t t0 = millis();
  while (time(nullptr) < 1700000000 && millis() - t0 < 20000) delay(250);
  const time_t now = time(nullptr);
  if (now < 1700000000) {
    Serial.println("[OTA] NTP FAILED - cannot validate HTTPS certificates without the date");
    return false;
  }
  struct tm tm_utc;
  gmtime_r(&now, &tm_utc);
  char buf[32];
  strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
  Serial.printf("[OTA] TIME_SYNCED %s\n", buf);
  return true;
}

void configure_tls(WiFiClientSecure* client) {
  if (g_cfg.tls_insecure) client->setInsecure();
  else client->setCACert(kCaRoots);
  client->setTimeout(kHttpTimeoutMs / 1000);
}

// GETs `url` (following GitHub's redirect to its asset host) and calls sink(chunk) per block.
template <typename Sink>
int https_get(const char* url, int32_t max_bytes, Sink sink, int32_t* total) {
  WiFiClientSecure client;
  configure_tls(&client);
  HTTPClient http;
  http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
  http.setTimeout(kHttpTimeoutMs);
  http.setUserAgent("tinymlops-esp32");
  if (!http.begin(client, url)) return -1;
  const int code = http.GET();
  if (code != HTTP_CODE_OK) {
    http.end();
    return code;
  }
  const int len = http.getSize();  // -1 when chunked
  if (len > max_bytes) {
    http.end();
    return -2;
  }
  WiFiClient* stream = http.getStreamPtr();
  uint8_t chunk[1024];
  int32_t got = 0;
  uint32_t last_data = millis();
  while (http.connected() && (len < 0 || got < len)) {
    const size_t avail = stream->available();
    if (avail) {
      const int n = stream->readBytes(chunk, avail > sizeof(chunk) ? sizeof(chunk) : avail);
      if (n <= 0) break;
      if (got + n > max_bytes) {
        http.end();
        return -2;
      }
      if (!sink(chunk, (size_t)n)) {
        http.end();
        return -3;
      }
      got += n;
      last_data = millis();
    } else if (millis() - last_data > kHttpTimeoutMs) {
      break;
    } else {
      delay(2);
    }
  }
  http.end();
  *total = got;
  return (len < 0 || got == len) ? HTTP_CODE_OK : -4;
}

// Streams url -> path while hashing; true only if size and SHA-256 both match.
bool download(const char* what, const char* url, const char* path, int32_t expect_size, const char* expect_sha) {
  File f = LittleFS.open(path, "w");
  if (!f) {
    Serial.printf("[OTA] DOWNLOAD %s FAILED: cannot open %s\n", what, path);
    return false;
  }
  Sha256 sha;
  int32_t total = 0;
  const uint32_t t0 = millis();
  const int code = https_get(url, expect_size, [&](const uint8_t* d, size_t n) {
    sha.update(d, n);
    return f.write(d, n) == n;
  }, &total);
  f.close();
  char hex[65];
  sha.final_hex(hex);
  if (code != HTTP_CODE_OK || total != expect_size) {
    Serial.printf("[OTA] DOWNLOAD %s FAILED: http=%d bytes=%ld/%ld\n", what, code, (long)total, (long)expect_size);
    return false;
  }
  const bool ok = strcmp(hex, expect_sha) == 0;
  Serial.printf("[OTA] DOWNLOAD %s %ld B in %lu ms -> %s SHA256=%s\n", what, (long)total,
                (unsigned long)(millis() - t0), ok ? "SHA_OK" : "SHA_MISMATCH", hex);
  return ok;
}

void check_once() {
  ++g_checks;
  Serial.printf("[OTA] CHECK #%ld manifest=%s\n", (long)g_checks, g_cfg.manifest_url);
  if (!wifi_up()) return result("ERROR", "wifi_unavailable", 0);
  if (!g_cfg.tls_insecure && !time_synced()) return result("ERROR", "ntp_failed", 0);

  // 1) manifest
  static char json[kManifestMax + 1];
  size_t jlen = 0;
  int32_t total = 0;
  const int code = https_get(g_cfg.manifest_url, kManifestMax, [&](const uint8_t* d, size_t n) {
    memcpy(json + jlen, d, n);
    jlen += n;
    return true;
  }, &total);
  json[jlen] = '\0';
  if (code != HTTP_CODE_OK) {
    char why[40];
    snprintf(why, sizeof(why), "manifest_http_%d", code);
    Serial.printf("[OTA] MANIFEST FAILED http=%d (no model published yet, or no internet)\n", code);
    return result("ERROR", why, 0);
  }
  Manifest m;
  const ManifestError me = parse_manifest(json, &m);
  if (me != ManifestError::kOk) return result("REJECTED", manifest_error_name(me), 0);
  Serial.printf("[OTA] MANIFEST version=%ld tag=%s model=%ld B sha=%.16s... int8_acc=%.2f%% min_fw=%s\n",
                (long)m.version, m.tag, (long)m.model_size, m.model_sha256, 100.f * m.int8_accuracy, m.min_firmware);

  // 2) decide
  const ActiveState cur = g_dep->state();
  const int32_t max_model = (int32_t)ESP.getMaxAllocHeap() - 64 * 1024;  // keep room for TLS buffers
  const OtaDecision d =
      decide_update(m, cur.version, cur.sha, g_cfg.firmware_version, g_cfg.supported_ops, max_model);
  Serial.printf("[OTA] DECISION %s (%s) current=%s v%ld\n",
                d.action == OtaAction::kUpdate ? "UPDATE" : d.action == OtaAction::kUpToDate ? "UP_TO_DATE" : "REJECT",
                d.reason, g_dep->model_name(), (long)cur.version);
  if (d.action == OtaAction::kUpToDate) return result("UP_TO_DATE", d.reason, m.version);
  if (d.action == OtaAction::kReject) return result("REJECTED", d.reason, m.version);

  // 3) download into the idle slot
  const char slot = ModelStore::other_slot(cur.slot);
  char mpath[32], vpath[32];
  ModelStore::model_path(slot, mpath, sizeof(mpath));
  ModelStore::vectors_path(slot, vpath, sizeof(vpath));
  Serial.printf("[OTA] DOWNLOADING %s into slot %c\n", m.tag, slot);
  blink(2, 80, 80);
  if (!download("model", m.model_url, mpath, m.model_size, m.model_sha256) ||
      !download("vectors", m.vectors_url, vpath, m.vectors_size, m.vectors_sha256)) {
    g_store->remove_slot(slot);
    blink(10, 50, 50);
    return result("REJECTED", "download_or_sha_failed", m.version);  // active model never touched
  }

  // 4) on-device self-test, then switch or roll back
  const CandidateResult c = g_dep->try_candidate(slot, m);
  if (c.activated) {
    Serial.printf("[OTA] ACTIVATED %s in slot %c: BIST %d/%d, bit-exact %d/%d, avg %.1f us\n", m.tag, slot,
                  c.bist.passed, c.bist.total, c.bist.bitexact, c.bist.total, c.bist.avg_us);
    g_dep->print_telemetry();
    blink(3, 200, 150);
    return result("UPDATED", c.reason, m.version);
  }
  Serial.printf("[OTA] ROLLBACK: candidate %s rejected (%s); still running %s\n", m.tag, c.reason, g_dep->model_name());
  g_store->remove_slot(slot);
  blink(10, 50, 50);
  result("ROLLED_BACK", c.reason, m.version);
}

void ota_task(void*) {
  for (;;) {
    check_once();
    // Sleep until the next interval, or until ota_trigger() wakes us.
    ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS((uint64_t)g_cfg.interval_s * 1000));
  }
}

}  // namespace

void ota_start(Deployment* deployment, ModelStore* store, const OtaConfig& cfg) {
  g_dep = deployment;
  g_store = store;
  g_cfg = cfg;
  Serial.printf("[OTA] ENABLED interval=%lus tls=%s manifest=%s\n", (unsigned long)cfg.interval_s,
                cfg.tls_insecure ? "INSECURE" : "verified", cfg.manifest_url);
  // The TLS handshake (certificate-chain signature checks) is CPU-bound for well over the task
  // watchdog's 5 s in the Wokwi simulator (CPU capped at ~8 MHz), starving core 0's idle task and
  // aborting the firmware. Stop the watchdog from checking IDLE0; the WiFi stack runs at a higher
  // priority on core 0, so it is unaffected.
  disableCore0WDT();
  // Core 0: the Arduino loop (serial commands) keeps running on core 1 during downloads.
  xTaskCreatePinnedToCore(ota_task, "ota", 16384, nullptr, 1, &g_task, 0);
}

void ota_trigger() {
  if (g_task) xTaskNotifyGive(g_task);
}

void ota_print_status() {
  Serial.printf("[OTA] STATUS last=%s reason=%s checks=%ld wifi=%s ip=%s\n", g_last_status, g_last_detail,
                (long)g_checks, WiFi.status() == WL_CONNECTED ? "connected" : "down",
                WiFi.status() == WL_CONNECTED ? WiFi.localIP().toString().c_str() : "-");
}

}  // namespace tinymlops
