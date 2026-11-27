// Model storage on the LittleFS data partition ("spiffs" in the default partition table):
//
//   /A/model.tflite  /A/vectors.bin      slot A
//   /B/model.tflite  /B/vectors.bin      slot B
//   /state.txt                           which model is active (factory | A | B) + its version/sha/tag
//
// The factory model compiled into the firmware is never overwritten, so the device can always
// fall back to it. Updates go to the slot that is NOT active, and /state.txt is replaced
// atomically (write temp + rename) only after the new model passed its on-device self-test.
#ifndef TINYMLOPS_MODEL_STORE_H_
#define TINYMLOPS_MODEL_STORE_H_

#include <stddef.h>
#include <stdint.h>

namespace tinymlops {

struct ActiveState {
  char slot = 'F';      // 'F' factory, 'A', 'B'
  int32_t version = 0;  // factory = 0
  char sha[65] = {0};
  char tag[40] = {0};
};

// Heap buffer whose data pointer is 16-byte aligned (TFLite Micro requirement).
struct AlignedBuffer {
  uint8_t* raw = nullptr;
  uint8_t* data = nullptr;
  size_t len = 0;
  bool alloc(size_t n);
  void release();
};

class ModelStore {
 public:
  bool begin();  // mounts LittleFS (formats it on first use)
  bool mounted() const { return mounted_; }
  bool read_state(ActiveState* out);
  bool write_state(const ActiveState& s);
  static char other_slot(char active) { return active == 'A' ? 'B' : 'A'; }
  static void model_path(char slot, char* out, size_t n);
  static void vectors_path(char slot, char* out, size_t n);
  bool read_file(const char* path, AlignedBuffer* buf);
  bool remove_slot(char slot);
  size_t total_bytes() const;
  size_t used_bytes() const;

 private:
  bool mounted_ = false;
};

}  // namespace tinymlops

#endif  // TINYMLOPS_MODEL_STORE_H_
