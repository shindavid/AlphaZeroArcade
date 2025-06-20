#pragma once

#include <mcts/Constants.hpp>
#include <util/CppUtil.hpp>

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
