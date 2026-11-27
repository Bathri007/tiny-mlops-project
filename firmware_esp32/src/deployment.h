// Which model is running, and the only code allowed to change it.
//
//   boot()           : run the model recorded in /state.txt; if its files are missing, its SHA-256
//                      no longer matches, or its self-test fails -> fall back to the factory model.
//   try_candidate()  : load a freshly downloaded slot, verify SHA-256, run the self-test with the
//                      vectors that came with it; switch only if it passes, otherwise put the
//                      previous model back (rollback). The previous model stays in RAM until then.
//
// All runtime access goes through a mutex, so the background OTA task can swap models while the
// serial command loop keeps answering INFER / SELFTEST.
#ifndef TINYMLOPS_DEPLOYMENT_H_
#define TINYMLOPS_DEPLOYMENT_H_

#include <stddef.h>
#include <stdint.h>

#include "manifest.h"
#include "model_runtime.h"
#include "model_store.h"
#include "vector_pack.h"

namespace tinymlops {

struct CandidateResult {
  bool activated = false;
  const char* reason = "";
  BistReport bist;
};

class Deployment {
 public:
  Deployment(ModelRuntime& rt, ModelStore& store) : rt_(rt), store_(store) {}

  void set_factory(const uint8_t* model, size_t len, const char* sha, const char* name, const VectorSet& vectors);
  void set_require_bitexact(bool b) { require_bitexact_ = b; }

  bool boot();  // prints the boot telemetry + BIST; returns true if the active model passed its BIST
  CandidateResult try_candidate(char slot, const Manifest& m);
  bool force_factory(bool persist = true);

  // Thread-safe wrappers used by the serial command loop.
  InferenceResult infer(const int8_t* image);
  BistReport run_bist(bool print_table);
  // Runs golden vector i of the ACTIVE model; returns false if i is out of range.
  bool selftest(uint16_t i, InferenceResult* r, uint8_t* label, bool* exact);
  uint16_t vector_count() const { return vectors_.count; }
  int num_classes() const { return vectors_.num_classes; }
  size_t image_size() const { return vectors_.image_size; }

  ActiveState state() const { return state_; }
  const char* runtime_sha() const { return runtime_sha_; }
  bool integrity_ok() const { return integrity_ok_; }
  const char* model_name() const;
  BistReport last_bist() const { return last_bist_; }
  void print_telemetry();

 private:
  bool load_slot(char slot, const char* expected_model_sha, const char* expected_vectors_sha,
                 AlignedBuffer* model, AlignedBuffer* vecs, VectorSet* vs, const char** reason);
  bool bist_ok(const BistReport& r) const;
  void lock();
  void unlock();

  ModelRuntime& rt_;
  ModelStore& store_;
  const uint8_t* factory_model_ = nullptr;
  size_t factory_len_ = 0;
  const char* factory_sha_ = "";
  const char* factory_name_ = "factory";
  VectorSet factory_vectors_;

  ActiveState state_;
  AlignedBuffer model_buf_;    // active slot model (unused for factory)
  AlignedBuffer vectors_buf_;  // active slot vectors
  VectorSet vectors_;
  char runtime_sha_[65] = {0};
  bool integrity_ok_ = false;
  bool require_bitexact_ = true;
  BistReport last_bist_;
  void* mutex_ = nullptr;
};

}  // namespace tinymlops

#endif  // TINYMLOPS_DEPLOYMENT_H_
