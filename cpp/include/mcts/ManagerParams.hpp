#pragma once

#include <core/concepts/Game.hpp>
#include <mcts/Constants.hpp>
#include <mcts/NNEvaluationServiceParams.hpp>
#include <util/CppUtil.hpp>
#include <util/Graph.hpp>

#include <boost/filesystem.hpp>

#include <cstdint>
#include <string>


namespace mcts {

template <core::concepts::Game Game, bool EnableStorage>
struct GraphVizPtr {
  GraphVizPtr(util::GraphViz<Game>* graph_viz = nullptr) {}
  bool operator==(const GraphVizPtr& other) const = default;
  util::GraphViz<Game>* get_graph_viz() const { return nullptr; }
};

template <core::concepts::Game Game>
struct GraphVizPtr<Game, true> {
  GraphVizPtr(util::GraphViz<Game>* graph_viz = nullptr) : graph_viz(graph_viz) {}
  bool operator==(const GraphVizPtr& other) const { return graph_viz == other.graph_viz; }
  util::GraphViz<Game>* get_graph_viz() const { return graph_viz; }

  util::GraphViz<Game>* graph_viz = nullptr;
};

static constexpr bool kStoreStates = IS_MACRO_ENABLED(STORE_STATES);

/*
 * ManagerParams pertains to a single mcts::Manager instance.
 *
 * By contrast, SearchParams pertains to each individual search() call.
 */
template <core::concepts::Game Game>
struct ManagerParams : public NNEvaluationServiceParams, public GraphVizPtr<Game, kStoreStates> {
  ManagerParams(mcts::Mode, util::GraphViz<Game>* graph_viz = nullptr);

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
};

}  // namespace mcts

#include <inline/mcts/ManagerParams.inl>
