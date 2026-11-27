#include "vector_pack.h"

#include <string.h>

namespace tinymlops {
namespace {

uint16_t rd_u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
uint32_t rd_u32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
float rd_f32(const uint8_t* p) {
  uint32_t u = rd_u32(p);
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

}  // namespace

const char* pack_error_name(PackError e) {
  switch (e) {
    case PackError::kOk: return "ok";
    case PackError::kTooShort: return "too_short";
    case PackError::kBadMagic: return "bad_magic";
    case PackError::kBadCounts: return "bad_counts";
    case PackError::kSizeMismatch: return "size_mismatch";
  }
  return "unknown";
}

size_t vector_pack_size(uint16_t count, uint16_t image_size, uint16_t num_classes) {
  return kVectorPackHeaderBytes + (size_t)count * (1 + num_classes + image_size);
}

PackError parse_vector_pack(const uint8_t* data, size_t len, VectorSet* out) {
  if (len < kVectorPackHeaderBytes) return PackError::kTooShort;
  if (memcmp(data, "TMV1", 4) != 0) return PackError::kBadMagic;
  VectorSet v;
  v.count = rd_u16(data + 4);
  v.image_size = rd_u16(data + 6);
  v.num_classes = rd_u16(data + 8);
  if (v.count == 0 || v.count > kVectorPackMaxCount || v.image_size == 0 || v.num_classes == 0) {
    return PackError::kBadCounts;
  }
  if (len != vector_pack_size(v.count, v.image_size, v.num_classes)) return PackError::kSizeMismatch;
  v.input_scale = rd_f32(data + 12);
  v.input_zero_point = (int32_t)rd_u32(data + 16);
  v.output_scale = rd_f32(data + 20);
  v.output_zero_point = (int32_t)rd_u32(data + 24);
  const uint8_t* p = data + kVectorPackHeaderBytes;
  v.labels = p;
  p += v.count;
  v.expected_logits = reinterpret_cast<const int8_t*>(p);
  p += (size_t)v.count * v.num_classes;
  v.images = reinterpret_cast<const int8_t*>(p);
  *out = v;
  return PackError::kOk;
}

}  // namespace tinymlops
