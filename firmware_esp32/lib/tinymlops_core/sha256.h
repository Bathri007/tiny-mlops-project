// Incremental SHA-256 (FIPS 180-4). Platform-independent: used by the firmware to
// verify models at rest and while streaming downloads, and compiled on the host by
// tests/test_firmware_core.py.
#ifndef TINYMLOPS_SHA256_H_
#define TINYMLOPS_SHA256_H_

#include <stddef.h>
#include <stdint.h>

namespace tinymlops {

class Sha256 {
 public:
  Sha256() { reset(); }
  void reset();
  void update(const uint8_t* data, size_t len);
  // Writes 64 lowercase hex chars + NUL into out_hex. The object must be reset() before reuse.
  void final_hex(char out_hex[65]);

 private:
  void block(const uint8_t blk[64]);
  uint32_t h_[8];
  uint8_t buf_[64];
  size_t buf_len_;
  uint64_t total_len_;
};

// One-shot helper.
void sha256_hex(const uint8_t* data, size_t len, char out_hex[65]);

}  // namespace tinymlops

#endif  // TINYMLOPS_SHA256_H_
