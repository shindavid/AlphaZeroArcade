#pragma once

#include <cstdint>
#include <string>

#include <boost/filesystem.hpp>

#include <mcts/Constants.hpp>
#include <util/CppUtil.hpp>

namespace mcts {

/*
 * Controls behavior of an NNEvaluationService.
 */
struct NNEvaluationServiceParams {
  auto make_options_description();
  bool operator==(const NNEvaluationServiceParams& other) const = default;

  std::string model_filename;
  std::string cuda_device = "cuda:0";
  int batch_size_limit = 216;
  int64_t nn_eval_timeout_ns = util::us_to_ns(250);
  size_t cache_size = 1048576;
};

}  // namespace mcts

#include <mcts/inl/NNEvaluationServiceParams.inl>
