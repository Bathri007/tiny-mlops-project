// Host-side driver for firmware_esp32/lib/tinymlops_core, compiled with g++ by
// tests/test_firmware_core.py so the exact C++ that runs on the ESP32 is unit-tested on a PC.
//
//   core_cli sha <file> <chunk>                         streaming SHA-256 (chunk = bytes per update)
//   core_cli pack <file>                                parse a TMV1 vector pack -> JSON summary
//   core_cli manifest <file> <cur_ver> <cur_sha> <fw> <ops> <max_bytes>   parse + decide
//   core_cli cmpver <a> <b>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "manifest.h"
#include "sha256.h"
#include "vector_pack.h"

using namespace tinymlops;

static std::vector<uint8_t> slurp(const char* path) {
  std::vector<uint8_t> d;
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
  uint8_t buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), f)) > 0) d.insert(d.end(), buf, buf + n);
  fclose(f);
  return d;
}

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string cmd = argv[1];
  if (cmd == "sha" && argc == 4) {
    const auto d = slurp(argv[2]);
    const size_t chunk = (size_t)atoi(argv[3]);
    Sha256 s;
    for (size_t i = 0; i < d.size(); i += chunk) s.update(d.data() + i, d.size() - i < chunk ? d.size() - i : chunk);
    char hex[65];
    s.final_hex(hex);
    printf("%s\n", hex);
    return 0;
  }
  if (cmd == "pack" && argc == 3) {
    const auto d = slurp(argv[2]);
    VectorSet v;
    const PackError e = parse_vector_pack(d.data(), d.size(), &v);
    if (e != PackError::kOk) { printf("{\"error\":\"%s\"}\n", pack_error_name(e)); return 0; }
    long img_sum = 0, logit_sum = 0;
    for (size_t i = 0; i < (size_t)v.count * v.image_size; ++i) img_sum += v.images[i];
    for (size_t i = 0; i < (size_t)v.count * v.num_classes; ++i) logit_sum += v.expected_logits[i];
    printf("{\"count\":%u,\"image_size\":%u,\"num_classes\":%u,\"input_scale\":%.9g,\"input_zero_point\":%d,"
           "\"output_scale\":%.9g,\"output_zero_point\":%d,\"labels\":[",
           v.count, v.image_size, v.num_classes, v.input_scale, v.input_zero_point, v.output_scale,
           v.output_zero_point);
    for (int i = 0; i < v.count; ++i) printf(i ? ",%u" : "%u", v.labels[i]);
    printf("],\"image_sum\":%ld,\"logit_sum\":%ld,\"last_pixel\":%d}\n", img_sum, logit_sum,
           v.image(v.count - 1)[v.image_size - 1]);
    return 0;
  }
  if (cmd == "manifest" && argc == 8) {
    auto d = slurp(argv[2]);
    d.push_back(0);
    Manifest m;
    const ManifestError e = parse_manifest(reinterpret_cast<const char*>(d.data()), &m);
    if (e != ManifestError::kOk) { printf("{\"parse\":\"%s\"}\n", manifest_error_name(e)); return 0; }
    const OtaDecision dec = decide_update(m, atoi(argv[3]), argv[4], argv[5], argv[6], atoi(argv[7]));
    const char* act = dec.action == OtaAction::kUpdate ? "update" : dec.action == OtaAction::kUpToDate ? "up_to_date" : "reject";
    printf("{\"parse\":\"ok\",\"version\":%d,\"tag\":\"%s\",\"model_size\":%d,\"model_sha256\":\"%s\","
           "\"model_url\":\"%s\",\"int8_accuracy\":%.6f,\"action\":\"%s\",\"reason\":\"%s\"}\n",
           (int)m.version, m.tag, (int)m.model_size, m.model_sha256, m.model_url, m.int8_accuracy, act, dec.reason);
    return 0;
  }
  if (cmd == "cmpver" && argc == 4) {
    printf("%d\n", compare_versions(argv[2], argv[3]));
    return 0;
  }
  return 2;
}
