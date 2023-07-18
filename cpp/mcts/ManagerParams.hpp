#pragma once

#include <cstdint>
#include <string>

#include <boost/filesystem.hpp>

#include <mcts/Constants.hpp>
#include <util/CppUtil.hpp>

namespace mcts {

/*
 * ManagerParams pertains to a single mcts::Manager instance.
 *
 * By contrast, SearchParams pertains to each individual search() call.
 */
struct ManagerParams {
  ManagerParams(mcts::Mode);

  auto make_options_description();
  bool operator==(const ManagerParams& other) const = default;

#ifdef PROFILE_MCTS
  boost::filesystem::path profiling_dir() const { return boost::filesystem::path(profiling_dir_str); }
#else  // PROFILE_MCTS
  boost::filesystem::path profiling_dir() const { return {}; }
#endif  // PROFILE_MCTS

  std::string model_filename;
  std::string cuda_device = "cuda:0";
  int num_search_threads = 8;
  int batch_size_limit = 216;
  bool enable_pondering = false;  // pondering = think during opponent's turn
  int pondering_tree_size_limit = 4096;
  int64_t nn_eval_timeout_ns = util::us_to_ns(250);
  size_t cache_size = 1048576;

  std::string root_softmax_temperature_str;
  float cPUCT = 1.1;
  float cFPU = 0.2;
  float dirichlet_mult = 0.25;
  float dirichlet_alpha_sum = 0.03 * 361;
  bool forced_playouts = true;
  bool enable_first_play_urgency = true;
  float k_forced = 2.0;
#ifdef PROFILE_MCTS
  std::string profiling_dir_str;
#endif  // PROFILE_MCTS
};

}  // namespace mcts

#include <mcts/inl/ManagerParams.inl>
