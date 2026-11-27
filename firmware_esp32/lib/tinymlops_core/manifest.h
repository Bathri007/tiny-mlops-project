// Model release manifest (manifest.json, written by ml_pipeline/package_model.py) and the
// pure decision logic of the over-the-air (OTA) model updater.
//
// The manifest is a FLAT JSON object of strings / integers / numbers, e.g.
//   {"schema":1,"version":57,"tag":"model-v57","model_url":"https://...","model_sha256":"...",
//    "model_size":18816,"vectors_url":"https://...","vectors_sha256":"...","vectors_size":7928,
//    "int8_accuracy":0.9825,"min_firmware":"2.0.0","ops":"CONV_2D,MAX_POOL_2D,RESHAPE,FULLY_CONNECTED"}
// A tiny flat reader is used instead of a JSON library so this logic compiles and is unit-tested
// on the host exactly as it runs on the ESP32.
#ifndef TINYMLOPS_MANIFEST_H_
#define TINYMLOPS_MANIFEST_H_

#include <stddef.h>
#include <stdint.h>

namespace tinymlops {

constexpr size_t kUrlMax = 256;
constexpr size_t kShaHex = 65;

struct Manifest {
  int schema = 0;
  int32_t version = 0;
  char tag[40] = {0};
  char model_url[kUrlMax] = {0};
  char model_sha256[kShaHex] = {0};
  int32_t model_size = 0;
  char vectors_url[kUrlMax] = {0};
  char vectors_sha256[kShaHex] = {0};
  int32_t vectors_size = 0;
  float int8_accuracy = 0.f;
  char min_firmware[16] = {0};
  char ops[160] = {0};
};

// Flat JSON field readers. Return false when the key is missing or malformed.
bool json_get_string(const char* json, const char* key, char* out, size_t out_len);
bool json_get_int(const char* json, const char* key, int32_t* out);
bool json_get_float(const char* json, const char* key, float* out);

enum class ManifestError { kOk = 0, kMissingField, kBadSchema, kBadSha, kBadSize, kBadUrl };
const char* manifest_error_name(ManifestError e);
ManifestError parse_manifest(const char* json, Manifest* out);

// "1.2.3" style compare: <0, 0, >0. Missing parts count as 0.
int compare_versions(const char* a, const char* b);

// True if every comma-separated op in `ops` appears in `supported` (also comma-separated).
bool ops_supported(const char* ops, const char* supported);

enum class OtaAction { kUpToDate = 0, kUpdate, kReject };
struct OtaDecision {
  OtaAction action;
  const char* reason;  // static string, never null
};

// Decides what to do with a fetched manifest.
//   current_version / current_sha : the model running now (factory = version 0)
//   firmware_version              : this firmware's semantic version
//   supported_ops                 : ops registered in the firmware's op resolver
//   max_model_bytes               : what the device can hold in RAM for a model
OtaDecision decide_update(const Manifest& m, int32_t current_version, const char* current_sha,
                          const char* firmware_version, const char* supported_ops, int32_t max_model_bytes);

}  // namespace tinymlops

#endif  // TINYMLOPS_MANIFEST_H_
