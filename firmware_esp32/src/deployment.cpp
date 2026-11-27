#include "deployment.h"

#include <Arduino.h>

#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "sha256.h"

namespace tinymlops {

void Deployment::lock() {
  if (!mutex_) mutex_ = xSemaphoreCreateRecursiveMutex();
  xSemaphoreTakeRecursive(static_cast<SemaphoreHandle_t>(mutex_), portMAX_DELAY);
}
void Deployment::unlock() { xSemaphoreGiveRecursive(static_cast<SemaphoreHandle_t>(mutex_)); }

void Deployment::set_factory(const uint8_t* model, size_t len, const char* sha, const char* name,
                             const VectorSet& vectors) {
  factory_model_ = model;
  factory_len_ = len;
  factory_sha_ = sha;
  factory_name_ = name;
  factory_vectors_ = vectors;
}

const char* Deployment::model_name() const { return state_.slot == 'F' ? factory_name_ : state_.tag; }

bool Deployment::bist_ok(const BistReport& r) const {
  return r.all_passed() && (!require_bitexact_ || r.bitexact == r.total);
}

bool Deployment::load_slot(char slot, const char* expected_model_sha, const char* expected_vectors_sha,
                           AlignedBuffer* model, AlignedBuffer* vecs, VectorSet* vs, const char** reason) {
  char path[32];
  ModelStore::model_path(slot, path, sizeof(path));
  if (!store_.read_file(path, model)) { *reason = "model_file_missing"; return false; }
  ModelStore::vectors_path(slot, path, sizeof(path));
  if (!store_.read_file(path, vecs)) { *reason = "vectors_file_missing"; model->release(); return false; }
  char sha[65];
  sha256_hex(model->data, model->len, sha);
  if (expected_model_sha && strcmp(sha, expected_model_sha) != 0) { *reason = "model_sha_mismatch"; goto fail; }
  if (expected_vectors_sha) {
    sha256_hex(vecs->data, vecs->len, sha);
    if (strcmp(sha, expected_vectors_sha) != 0) { *reason = "vectors_sha_mismatch"; goto fail; }
  }
  if (parse_vector_pack(vecs->data, vecs->len, vs) != PackError::kOk) { *reason = "vectors_corrupt"; goto fail; }
  return true;
fail:
  model->release();
  vecs->release();
  return false;
}

bool Deployment::force_factory(bool persist) {
  lock();
  rt_.begin(factory_model_, factory_len_);
  model_buf_.release();
  vectors_buf_.release();
  vectors_ = factory_vectors_;
  state_ = ActiveState();
  strlcpy(state_.sha, factory_sha_, sizeof(state_.sha));
  strlcpy(state_.tag, "factory", sizeof(state_.tag));
  sha256_hex(factory_model_, factory_len_, runtime_sha_);
  integrity_ok_ = strcmp(runtime_sha_, factory_sha_) == 0;
  if (persist) store_.write_state(state_);
  const bool ok = rt_.ready();
  unlock();
  return ok;
}

bool Deployment::boot() {
  lock();
  ActiveState saved;
  store_.read_state(&saved);
  bool on_slot = false;
  if (saved.slot == 'A' || saved.slot == 'B') {
    AlignedBuffer m, v;
    VectorSet vs;
    const char* reason = "";
    if (load_slot(saved.slot, saved.sha, nullptr, &m, &v, &vs, &reason) && rt_.begin(m.data, m.len) &&
        rt_.vectors_match(vs)) {
      BistReport r = rt_.run_bist(vs, false);
      if (bist_ok(r)) {
        model_buf_ = m;
        vectors_buf_ = v;
        vectors_ = vs;
        state_ = saved;
        strlcpy(runtime_sha_, saved.sha, sizeof(runtime_sha_));
        integrity_ok_ = true;
        on_slot = true;
      } else {
        reason = "boot_bist_failed";
      }
    } else if (!rt_.ready() && *reason == '\0') {
      reason = rt_.last_error();
    }
    if (!on_slot) {
      Serial.printf("[BOOT] slot %c (%s) unusable: %s -> FALLBACK to factory model\n", saved.slot, saved.tag,
                    *reason ? reason : "vectors_not_for_model");
      m.release();
      v.release();
    }
  }
  if (!on_slot) force_factory(/*persist=*/saved.slot != 'F');  // only rewrite after a fallback
  print_telemetry();
  if (!rt_.ready()) {
    Serial.printf("[TINYMLOPS_FATAL] model bring-up failed: %s\n", rt_.last_error());
    unlock();
    return false;
  }
  last_bist_ = rt_.run_bist(vectors_, true);
  unlock();
  return bist_ok(last_bist_);
}

CandidateResult Deployment::try_candidate(char slot, const Manifest& m) {
  CandidateResult res;
  AlignedBuffer cm, cv;
  VectorSet cvs;
  if (!load_slot(slot, m.model_sha256, m.vectors_sha256, &cm, &cv, &cvs, &res.reason)) return res;

  lock();
  // Previous model stays in model_buf_ (or flash for factory) until the candidate has passed.
  const uint8_t* prev_model = state_.slot == 'F' ? factory_model_ : model_buf_.data;
  const size_t prev_len = state_.slot == 'F' ? factory_len_ : model_buf_.len;

  if (!rt_.begin(cm.data, cm.len)) {
    res.reason = "candidate_alloc_failed";
  } else if (!rt_.vectors_match(cvs)) {
    res.reason = "vectors_not_for_model";
  } else {
    Serial.printf("[OTA] self-testing candidate %s on device ...\n", m.tag);
    res.bist = rt_.run_bist(cvs, true);
    if (bist_ok(res.bist)) {
      ActiveState next;
      next.slot = slot;
      next.version = m.version;
      strlcpy(next.sha, m.model_sha256, sizeof(next.sha));
      strlcpy(next.tag, m.tag, sizeof(next.tag));
      if (store_.write_state(next)) {
        model_buf_.release();
        vectors_buf_.release();
        model_buf_ = cm;
        vectors_buf_ = cv;
        vectors_ = cvs;
        state_ = next;
        strlcpy(runtime_sha_, m.model_sha256, sizeof(runtime_sha_));
        integrity_ok_ = true;
        last_bist_ = res.bist;
        res.activated = true;
        res.reason = "self_test_passed";
        unlock();
        return res;
      }
      res.reason = "state_write_failed";
    } else {
      res.reason = res.bist.all_passed() ? "bist_not_bitexact" : "bist_failed";
    }
  }
  // Rollback: put the previous model back exactly as it was.
  rt_.begin(prev_model, prev_len);
  unlock();
  cm.release();
  cv.release();
  return res;
}

InferenceResult Deployment::infer(const int8_t* image) {
  lock();
  InferenceResult r = rt_.infer(image);
  unlock();
  return r;
}

BistReport Deployment::run_bist(bool print_table) {
  lock();
  last_bist_ = rt_.run_bist(vectors_, print_table);
  BistReport r = last_bist_;
  unlock();
  return r;
}

bool Deployment::selftest(uint16_t i, InferenceResult* r, uint8_t* label, bool* exact) {
  lock();
  const bool in_range = i < vectors_.count;
  if (in_range) {
    *r = rt_.infer(vectors_.image(i));
    *label = vectors_.labels[i];
    *exact = r->ok && memcmp(r->logits, vectors_.logits(i), vectors_.num_classes) == 0;
  }
  unlock();
  return in_range;
}

void Deployment::print_telemetry() {
  lock();
  Serial.printf("[TELEMETRY] MODEL_NAME=%s\n", model_name());
  Serial.printf("[TELEMETRY] MODEL_SLOT=%s MODEL_VERSION=%ld\n",
                state_.slot == 'F' ? "factory" : (state_.slot == 'A' ? "A" : "B"), (long)state_.version);
  Serial.printf("[TELEMETRY] MODEL_SHA256_EMBEDDED=%s\n", state_.sha);
  Serial.printf("[TELEMETRY] MODEL_SHA256_RUNTIME =%s\n", runtime_sha_);
  Serial.printf("[TELEMETRY] MODEL_INTEGRITY=%s\n", integrity_ok_ ? "OK" : "MISMATCH");
  Serial.printf("[TELEMETRY] STORAGE_USED=%u STORAGE_TOTAL=%u\n", (unsigned)store_.used_bytes(),
                (unsigned)store_.total_bytes());
  rt_.print_tensor_info();
  unlock();
}

}  // namespace tinymlops
