#pragma once

namespace mcts {

/*
 * SearchParams pertain to a single call to mcts::Manager::search(). Even given a single
 * mcts::Manager instance, different search() calls can have different SearchParams. For instance,
 * for KataGo, there are "fast" searches and "full" searches, which differ in their tree_size_limit
 * and dirchlet settings.
 *
 * By contrast, mcts::Manager::Params pertains to a single mcts::Manager instance.
 */
struct SearchParams {
  static SearchParams make_pondering_params(int limit) { return SearchParams{limit, true, true}; }

  int tree_size_limit = 100;
  bool disable_exploration = false;
  bool ponder = false;
};

}  // namespace mcts
