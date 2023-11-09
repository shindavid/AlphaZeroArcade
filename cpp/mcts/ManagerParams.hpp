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
  boost::filesystem::path profiling_dir() const {
    return boost::filesystem::path(profiling_dir_str);
  }
#else   // PROFILE_MCTS
  boost::filesystem::path profiling_dir() const { return {}; }
#endif  // PROFILE_MCTS

  std::string model_filename;
  std::string cuda_device = "cuda:0";
  int num_search_threads = 256;
  int batch_size_limit = 216;
  bool enable_pondering = false;  // pondering = think during opponent's turn
  int pondering_tree_size_limit = 4096;
  int64_t nn_eval_timeout_ns = util::us_to_ns(250);
  size_t cache_size = 1048576;

  std::string root_softmax_temperature_str;
  float cPUCT = 1.1;
  float cFPU = 0.2;
  float dirichlet_mult = 0.25;

  /*
   * For dirichlet noise, we use a uniform alpha = dirichlet_alpha_factor / sqrt(num_actions).
   */
  float dirichlet_alpha_factor = 0.57;  // ~= .03 * sqrt(361) to match AlphaGo
  bool forced_playouts = true;
  bool enable_first_play_urgency = true;
  float k_forced = 2.0;

  /*
   * These bools control both MCTS dynamics and the zeroing out of the MCTS counts exported to the
   * player (which in turn is exported as a policy training target).
   *
   * By default, in training mode, we set exploit_proven_winners to false. This is based on
   * empirical evidence that the exploitative behavior slows down learning. One reason this may be
   * the case is that when we exploit proven winners, we stop exploring other moves, some of which
   * may also be winners. This leads to policy training targets where one good move is arbitrarily
   * given all the weight and where other good moves are zeroed out. The neural network evidently
   * has trouble learning in the presence of such arbitrary masks.
   */
  bool exploit_proven_winners = true;
  bool avoid_proven_losers = true;

#ifdef PROFILE_MCTS
  std::string profiling_dir_str;
#endif  // PROFILE_MCTS
};

}  // namespace mcts

#include <mcts/inl/ManagerParams.inl>
