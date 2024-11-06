#pragma once

#include <core/concepts/Game.hpp>
#include <mcts/Constants.hpp>
#include <mcts/NNEvaluationServiceParams.hpp>
#include <util/CppUtil.hpp>

#include <boost/filesystem.hpp>

#include <cstdint>
#include <string>

namespace mcts {

/*
 * ManagerParams pertains to a single mcts::Manager instance.
 *
 * By contrast, SearchParams pertains to each individual search() call.
 */
template <core::concepts::Game Game>
struct ManagerParams : public NNEvaluationServiceParams {
  ManagerParams(mcts::Mode);

  auto make_options_description();
  bool operator==(const ManagerParams& other) const = default;

  int num_search_threads = 1;
  bool apply_random_symmetries = true;
  bool enable_pondering = false;  // pondering = think during opponent's turn
  int pondering_tree_size_limit = 4096;

  float starting_root_softmax_temperature = 1.4;
  float ending_root_softmax_temperature = 1.1;
  float root_softmax_temperature_half_life = 0.5 * Game::Constants::kOpeningLength;
  float cPUCT = 1.1;
  float cFPU = 0.2;
  float dirichlet_mult = 0.25;

  /*
   * For dirichlet noise, we use a uniform alpha = dirichlet_alpha_factor / sqrt(num_actions).
   */
  float dirichlet_alpha_factor = 0.57;  // ~= .03 * sqrt(361) to match AlphaGo
  bool forced_playouts = true;
  bool enable_first_play_urgency = false;
  float k_forced = 2.0;

  /*
   * These bools control both MCTS dynamics and the zeroing out of the MCTS counts exported to the
   * player (which in turn is exported as a policy training target).
   */
  bool exploit_proven_winners = true;
  bool avoid_proven_losers = true;

  /*
   * If true, we forcibly evaluate all children of root nodes. This is needed in training mode to
   * create action-value targets.
   */
  bool force_evaluate_all_root_children = false;

  /*
   * When we use a neural network to evaluate a position, we first apply a random symmetry to the
   * position.
   *
   * If incorporate_sym_into_cache_key is true, then this sym is part of the cache key.
   *
   * The benefit of incorporating sym into the cache key is that when multiple games are played,
   * those games are independent. The downside is that we get less cache hits, hurting game
   * throughput.
   *
   * Based on empirical testing, we find that in self-play, it's better not to incorporate sym
   * into the cache key, to maximize game throughput. The downside is mitigated by the fact that
   * the cache is cleared on each generation, leading to partial-independence. In contrast, for
   * rating games, we incorporate sym into the cache key, to ensure the games are truly
   * independent, in order to get more accurate ratings.
   */
  bool incorporate_sym_into_cache_key = true;
};

}  // namespace mcts

#include <inline/mcts/ManagerParams.inl>
