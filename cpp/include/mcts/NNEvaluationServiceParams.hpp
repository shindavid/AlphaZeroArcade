#pragma once

#include <boost/filesystem.hpp>

#include <cstdint>
#include <string>

namespace mcts {

/*
 * Controls behavior of an NNEvaluationService.
 */
struct NNEvaluationServiceParams {
  auto make_options_description();
  bool operator==(const NNEvaluationServiceParams& other) const = default;

  std::string model_filename;
  bool no_model = false;
  std::string cuda_device = "cuda:0";
  int num_pipelines = 2;
  size_t cache_size = 1048576;

  int batch_size = 256;
  uint64_t engine_build_workspace_size_in_bytes = 1 << 28;  // 256 MB
  std::string engine_build_precision = "FP16";
};

}  // namespace mcts

#include "inline/mcts/NNEvaluationServiceParams.inl"
