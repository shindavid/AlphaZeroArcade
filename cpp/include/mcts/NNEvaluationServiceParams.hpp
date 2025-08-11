#pragma once

#include <boost/filesystem.hpp>

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
};

}  // namespace mcts

#include "inline/mcts/NNEvaluationServiceParams.inl"
