#pragma once

#include "search/Constants.hpp"
#include "nnet/NNEvaluationServiceParams.hpp"
#include "search/SearchParams.hpp"

#include <boost/filesystem.hpp>

namespace search {

/*
 * ManagerParams pertains to a single search::Manager instance.
 *
 * By contrast, SearchParams pertains to each individual search() call.
 */
template <typename Traits>
struct ManagerParams : public nnet::NNEvaluationServiceParams {
  using Game = Traits::Game;

  ManagerParams(Mode);

  search::SearchParams pondering_search_params() const {
    return search::SearchParams::make_pondering_params(pondering_tree_size_limit);
  }

  auto make_options_description();
  bool operator==(const ManagerParams& other) const = default;

  int num_search_threads = 1;
  bool apply_random_symmetries = true;
  bool enable_pondering = false;  // pondering = think during opponent's turn
  int pondering_tree_size_limit = 4096;

  float starting_root_softmax_temperature = 1.4;
  float ending_root_softmax_temperature = 1.1;
  float root_softmax_temperature_half_life = 0.5 * Game::MctsConfiguration::kOpeningLength;
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
   * @dshin performed ad-hoc tests in 2024 that showed that setting this to false in evaluation
   * games contributed unacceptably large noise in rating evaluations. Based on that, we set it to
   * true for kCompetition mode.
   *
   * In August 2025, @lichensongs performed further ad-hoc tests in the games of hex and Othello.
   * These tests showed that in hex, incorporating sym into the cache key significantly decreased
   * variance in the the skill-curve. Furthermore, though the cache hit rate was indeed lower, the
   * overall skill progression with respect to self-play-time actually improved, for reasons we
   * don't yet completely understand. In Othello, we observed a slight reduction in skill-curve
   * variance, and no visible change in skill progression with respect to self-play-time.
   *
   * Skill-curve variance reduction is a worthy goal, as it allows us to perform A/B tests more
   * effectively.
   *
   * Based on these tests, we decided to set this bool to true in both modes.
   */
  bool incorporate_sym_into_cache_key = true;
};

}  // namespace search

#include "inline/search/ManagerParams.inl"
