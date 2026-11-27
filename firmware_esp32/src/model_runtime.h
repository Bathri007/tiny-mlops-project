// TFLite Micro runtime that can be re-pointed at a different model at run time (factory model in
// flash, or a model loaded from a LittleFS slot), plus the built-in self-test (BIST).
#ifndef TINYMLOPS_MODEL_RUNTIME_H_
#define TINYMLOPS_MODEL_RUNTIME_H_

#include <stddef.h>
#include <stdint.h>

#include "vector_pack.h"

namespace tinymlops {

constexpr int kMaxClasses = 16;

struct InferenceResult {
  bool ok = false;
  int pred = -1;
  int8_t score = 0;        // raw int8 logit of the winning class
  float confidence = 0.f;  // softmax(dequantized logits)[pred], percent
  int64_t latency_us = 0;  // Invoke() wall time
  int num_classes = 0;
  int8_t logits[kMaxClasses] = {0};
};

struct BistReport {
  int total = 0;
  int passed = 0;
  int bitexact = 0;
  double avg_us = 0;
  int64_t min_us = 0;
  int64_t max_us = 0;
  size_t arena_bytes = 0;
  bool all_passed() const { return total > 0 && passed == total; }
};

class ModelRuntime {
 public:
  // (Re)builds the interpreter for `model` (16-byte aligned, must outlive the runtime's use of it).
  // On failure, ready() is false and last_error() says why.
  bool begin(const uint8_t* model, size_t len);
  void end();
  bool ready() const { return ready_; }
  const char* last_error() const { return error_; }

  // Checks that a vector set was produced for this model (shapes + input/output quantization).
  bool vectors_match(const VectorSet& v) const;

  InferenceResult infer(const int8_t* image);
  BistReport run_bist(const VectorSet& v, bool print_table);

  size_t arena_used() const;
  size_t arena_size() const;
  size_t input_bytes() const;
  int num_classes() const;
  float input_scale() const;
  int input_zero_point() const;
  float output_scale() const;
  int output_zero_point() const;
  void print_tensor_info() const;

 private:
  bool ready_ = false;
  const char* error_ = "not started";
};

}  // namespace tinymlops

#endif  // TINYMLOPS_MODEL_RUNTIME_H_
