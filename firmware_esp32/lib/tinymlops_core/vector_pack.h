// Golden test vectors that travel with every model, so the device can self-test a model it
// just downloaded before switching to it. Two sources share one view type:
//   * the factory vectors compiled into the firmware (test_vectors.h), and
//   * a "TMV1" binary pack downloaded next to the model (written by ml_pipeline/package_model.py).
//
// TMV1 layout (little endian, no padding):
//   offset size  field
//   0      4     magic "TMV1"
//   4      2     count          (number of vectors, 1..64)
//   6      2     image_size     (int8 values per image, e.g. 784)
//   8      2     num_classes    (e.g. 10)
//   10     2     reserved (0)
//   12     4     input_scale    (float32)
//   16     4     input_zero_point  (int32)
//   20     4     output_scale   (float32)
//   24     4     output_zero_point (int32)
//   28     count                          labels      (uint8)
//   ..     count * num_classes            exp. logits (int8, host TFLite reference output)
//   ..     count * image_size             images      (int8, already quantized)
#ifndef TINYMLOPS_VECTOR_PACK_H_
#define TINYMLOPS_VECTOR_PACK_H_

#include <stddef.h>
#include <stdint.h>

namespace tinymlops {

constexpr size_t kVectorPackHeaderBytes = 28;
constexpr uint16_t kVectorPackMaxCount = 64;

struct VectorSet {
  uint16_t count = 0;
  uint16_t image_size = 0;
  uint16_t num_classes = 0;
  float input_scale = 0.f;
  int32_t input_zero_point = 0;
  float output_scale = 0.f;
  int32_t output_zero_point = 0;
  const uint8_t* labels = nullptr;          // [count]
  const int8_t* expected_logits = nullptr;  // [count][num_classes]
  const int8_t* images = nullptr;           // [count][image_size]

  const int8_t* image(uint16_t i) const { return images + (size_t)i * image_size; }
  const int8_t* logits(uint16_t i) const { return expected_logits + (size_t)i * num_classes; }
};

enum class PackError { kOk = 0, kTooShort, kBadMagic, kBadCounts, kSizeMismatch };
const char* pack_error_name(PackError e);

// Size in bytes of a TMV1 pack with these dimensions.
size_t vector_pack_size(uint16_t count, uint16_t image_size, uint16_t num_classes);

// Parses a pack in place: `out` points into `data`, which must outlive it.
PackError parse_vector_pack(const uint8_t* data, size_t len, VectorSet* out);

}  // namespace tinymlops

#endif  // TINYMLOPS_VECTOR_PACK_H_
