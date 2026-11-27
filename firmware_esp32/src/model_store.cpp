#include "model_store.h"

#include <Arduino.h>
#include <FS.h>
#include <LittleFS.h>

#include <stdlib.h>
#include <string.h>

namespace tinymlops {

bool AlignedBuffer::alloc(size_t n) {
  release();
  raw = static_cast<uint8_t*>(malloc(n + 16));
  if (!raw) return false;
  data = reinterpret_cast<uint8_t*>(((uintptr_t)raw + 15) & ~(uintptr_t)15);
  len = n;
  return true;
}

void AlignedBuffer::release() {
  free(raw);
  raw = data = nullptr;
  len = 0;
}

bool ModelStore::begin() {
  mounted_ = LittleFS.begin(/*formatOnFail=*/true);
  if (mounted_) {
    LittleFS.mkdir("/A");
    LittleFS.mkdir("/B");
  }
  return mounted_;
}

void ModelStore::model_path(char slot, char* out, size_t n) { snprintf(out, n, "/%c/model.tflite", slot); }
void ModelStore::vectors_path(char slot, char* out, size_t n) { snprintf(out, n, "/%c/vectors.bin", slot); }

size_t ModelStore::total_bytes() const { return mounted_ ? LittleFS.totalBytes() : 0; }
size_t ModelStore::used_bytes() const { return mounted_ ? LittleFS.usedBytes() : 0; }

bool ModelStore::read_state(ActiveState* out) {
  ActiveState s;
  if (!mounted_ || !LittleFS.exists("/state.txt")) {
    *out = s;  // factory
    return false;
  }
  File f = LittleFS.open("/state.txt", "r");
  while (f && f.available()) {
    String line = f.readStringUntil('\n');
    line.trim();
    const int eq = line.indexOf('=');
    if (eq <= 0) continue;
    const String k = line.substring(0, eq), v = line.substring(eq + 1);
    if (k == "slot" && v.length() == 1) s.slot = v[0];
    else if (k == "version") s.version = v.toInt();
    else if (k == "sha") strlcpy(s.sha, v.c_str(), sizeof(s.sha));
    else if (k == "tag") strlcpy(s.tag, v.c_str(), sizeof(s.tag));
  }
  f.close();
  if (s.slot != 'A' && s.slot != 'B') s = ActiveState();
  *out = s;
  return true;
}

bool ModelStore::write_state(const ActiveState& s) {
  if (!mounted_) return false;
  File f = LittleFS.open("/state.tmp", "w");
  if (!f) return false;
  f.printf("slot=%c\nversion=%ld\nsha=%s\ntag=%s\n", s.slot, (long)s.version, s.sha, s.tag);
  f.close();
  LittleFS.remove("/state.txt");
  return LittleFS.rename("/state.tmp", "/state.txt");
}

bool ModelStore::read_file(const char* path, AlignedBuffer* buf) {
  if (!mounted_ || !LittleFS.exists(path)) return false;
  File f = LittleFS.open(path, "r");
  if (!f) return false;
  const size_t n = f.size();
  if (n == 0 || !buf->alloc(n)) {
    f.close();
    return false;
  }
  const size_t got = f.read(buf->data, n);
  f.close();
  if (got != n) {
    buf->release();
    return false;
  }
  return true;
}

bool ModelStore::remove_slot(char slot) {
  char p[32];
  model_path(slot, p, sizeof(p));
  LittleFS.remove(p);
  vectors_path(slot, p, sizeof(p));
  LittleFS.remove(p);
  return true;
}

}  // namespace tinymlops
