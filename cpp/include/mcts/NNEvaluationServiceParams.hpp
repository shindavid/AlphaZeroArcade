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
  bool no_model = false;
  std::string cuda_device = "cuda:0";
  int batch_size_limit = 512;
  int64_t nn_eval_timeout_ns = util::ms_to_ns(1);
  size_t cache_size = 1048576;

#ifdef PROFILE_MCTS
  std::string profiling_dir_str;

  boost::filesystem::path profiling_dir() const {
    return boost::filesystem::path(profiling_dir_str);
  }
#else   // PROFILE_MCTS
  boost::filesystem::path profiling_dir() const { return {}; }
#endif  // PROFILE_MCTS

};

}  // namespace mcts

#include <inline/mcts/NNEvaluationServiceParams.inl>
