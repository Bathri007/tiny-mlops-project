#include "manifest.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

namespace tinymlops {
namespace {

// Returns a pointer to the first char of the value for "key", or nullptr. Only matches a key
// position (preceded by '{' or ','), so a key name inside a string value is never matched.
const char* find_value(const char* json, const char* key) {
  const size_t klen = strlen(key);
  for (const char* p = strchr(json, '"'); p; p = strchr(p + 1, '"')) {
    if (strncmp(p + 1, key, klen) != 0 || p[1 + klen] != '"') continue;
    const char* back = p - 1;
    while (back >= json && isspace((unsigned char)*back)) --back;
    if (back < json || (*back != '{' && *back != ',')) continue;
    const char* q = p + 2 + klen;
    while (isspace((unsigned char)*q)) ++q;
    if (*q != ':') continue;
    ++q;
    while (isspace((unsigned char)*q)) ++q;
    return q;
  }
  return nullptr;
}

bool is_sha_hex(const char* s) {
  if (strlen(s) != 64) return false;
  for (int i = 0; i < 64; ++i) {
    if (!isdigit((unsigned char)s[i]) && !(s[i] >= 'a' && s[i] <= 'f')) return false;
  }
  return true;
}

bool in_list(const char* item, size_t item_len, const char* list) {
  const char* p = list;
  while (*p) {
    while (*p == ' ' || *p == ',') ++p;
    const char* end = p;
    while (*end && *end != ',') ++end;
    const char* trim = end;
    while (trim > p && trim[-1] == ' ') --trim;
    if ((size_t)(trim - p) == item_len && strncmp(p, item, item_len) == 0) return true;
    p = end;
  }
  return false;
}

}  // namespace

bool json_get_string(const char* json, const char* key, char* out, size_t out_len) {
  const char* v = find_value(json, key);
  if (!v || *v != '"' || out_len == 0) return false;
  ++v;
  size_t n = 0;
  while (*v && *v != '"') {
    char c = *v++;
    if (c == '\\') {
      c = *v++;
      if (c == 'n') c = '\n';
      else if (c == 't') c = '\t';
      else if (c == 'u' || c == '\0') return false;  // \uXXXX not needed in our manifests
    }
    if (n + 1 >= out_len) return false;  // would truncate
    out[n++] = c;
  }
  if (*v != '"') return false;
  out[n] = '\0';
  return true;
}

bool json_get_int(const char* json, const char* key, int32_t* out) {
  const char* v = find_value(json, key);
  if (!v) return false;
  char* end = nullptr;
  long x = strtol(v, &end, 10);
  if (end == v || *end == '.' || *end == 'e' || *end == 'E') return false;
  *out = (int32_t)x;
  return true;
}

bool json_get_float(const char* json, const char* key, float* out) {
  const char* v = find_value(json, key);
  if (!v) return false;
  char* end = nullptr;
  float x = strtof(v, &end);
  if (end == v) return false;
  *out = x;
  return true;
}

const char* manifest_error_name(ManifestError e) {
  switch (e) {
    case ManifestError::kOk: return "ok";
    case ManifestError::kMissingField: return "missing_field";
    case ManifestError::kBadSchema: return "bad_schema";
    case ManifestError::kBadSha: return "bad_sha";
    case ManifestError::kBadSize: return "bad_size";
    case ManifestError::kBadUrl: return "bad_url";
  }
  return "unknown";
}

ManifestError parse_manifest(const char* json, Manifest* out) {
  Manifest m;
  int32_t schema = 0;
  if (!json_get_int(json, "schema", &schema)) return ManifestError::kMissingField;
  if (schema != 1) return ManifestError::kBadSchema;
  m.schema = schema;
  if (!json_get_int(json, "version", &m.version) ||
      !json_get_string(json, "tag", m.tag, sizeof(m.tag)) ||
      !json_get_string(json, "model_url", m.model_url, sizeof(m.model_url)) ||
      !json_get_string(json, "model_sha256", m.model_sha256, sizeof(m.model_sha256)) ||
      !json_get_int(json, "model_size", &m.model_size) ||
      !json_get_string(json, "vectors_url", m.vectors_url, sizeof(m.vectors_url)) ||
      !json_get_string(json, "vectors_sha256", m.vectors_sha256, sizeof(m.vectors_sha256)) ||
      !json_get_int(json, "vectors_size", &m.vectors_size) ||
      !json_get_string(json, "min_firmware", m.min_firmware, sizeof(m.min_firmware)) ||
      !json_get_string(json, "ops", m.ops, sizeof(m.ops))) {
    return ManifestError::kMissingField;
  }
  json_get_float(json, "int8_accuracy", &m.int8_accuracy);  // informational, optional
  if (!is_sha_hex(m.model_sha256) || !is_sha_hex(m.vectors_sha256)) return ManifestError::kBadSha;
  if (m.version <= 0 || m.model_size <= 0 || m.vectors_size <= 0) return ManifestError::kBadSize;
  if (strncmp(m.model_url, "https://", 8) != 0 || strncmp(m.vectors_url, "https://", 8) != 0) {
    return ManifestError::kBadUrl;
  }
  *out = m;
  return ManifestError::kOk;
}

int compare_versions(const char* a, const char* b) {
  for (int part = 0; part < 4; ++part) {
    long x = 0, y = 0;
    if (*a) x = strtol(a, const_cast<char**>(&a), 10);
    if (*b) y = strtol(b, const_cast<char**>(&b), 10);
    if (x != y) return x < y ? -1 : 1;
    if (*a == '.') ++a;
    if (*b == '.') ++b;
  }
  return 0;
}

bool ops_supported(const char* ops, const char* supported) {
  const char* p = ops;
  while (*p) {
    while (*p == ' ' || *p == ',') ++p;
    if (!*p) break;
    const char* end = p;
    while (*end && *end != ',') ++end;
    const char* trim = end;
    while (trim > p && trim[-1] == ' ') --trim;
    if (!in_list(p, (size_t)(trim - p), supported)) return false;
    p = end;
  }
  return true;
}

OtaDecision decide_update(const Manifest& m, int32_t current_version, const char* current_sha,
                          const char* firmware_version, const char* supported_ops, int32_t max_model_bytes) {
  if (compare_versions(firmware_version, m.min_firmware) < 0) return {OtaAction::kReject, "firmware_too_old"};
  if (!ops_supported(m.ops, supported_ops)) return {OtaAction::kReject, "unsupported_ops"};
  if (m.model_size > max_model_bytes) return {OtaAction::kReject, "model_too_large"};
  if (current_sha && strcmp(m.model_sha256, current_sha) == 0) return {OtaAction::kUpToDate, "same_model"};
  if (m.version <= current_version) return {OtaAction::kUpToDate, "not_newer"};
  return {OtaAction::kUpdate, "newer_version"};
}

}  // namespace tinymlops
