// Over-the-air MODEL updater (the firmware itself is not replaced).
//
// Background FreeRTOS task:  WiFi -> NTP time -> GET manifest.json (HTTPS, pinned root CAs)
//   -> decide_update() -> download model + vectors into the idle slot, SHA-256 checked while
//   streaming -> Deployment::try_candidate() (on-device self-test) -> switch or roll back.
// Runs at boot, every TINYMLOPS_OTA_INTERVAL_S seconds, and on demand (serial "OTA" command or the
// BOOT button). Every step prints a machine-parseable [OTA] line; each check ends with exactly one
//   [TINYMLOPS_OTA_RESULT] STATUS=<UPDATED|UP_TO_DATE|ROLLED_BACK|REJECTED|ERROR> ...
#ifndef TINYMLOPS_OTA_AGENT_H_
#define TINYMLOPS_OTA_AGENT_H_

#include "deployment.h"

namespace tinymlops {

struct OtaConfig {
  const char* wifi_ssid;
  const char* wifi_password;
  const char* manifest_url;
  const char* firmware_version;
  const char* supported_ops;
  uint32_t interval_s;
  bool tls_insecure;  // debugging only: skip certificate validation (SHA-256 checks still apply)
  int led_pin;
};

void ota_start(Deployment* deployment, ModelStore* store, const OtaConfig& cfg);
void ota_trigger();       // ask the task to check now (non-blocking)
void ota_print_status();  // last result, WiFi state

}  // namespace tinymlops

#endif  // TINYMLOPS_OTA_AGENT_H_
