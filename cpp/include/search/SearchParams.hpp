#pragma once

namespace search {

/*
 * SearchParams pertain to a single call to search::Manager::search(). Even given a single
 * search::Manager instance, different search() calls can have different SearchParams. For instance,
 * for KataGo, there are "fast" searches and "full" searches, which differ in their tree_size_limit
 * and dirchlet settings.
 *
 * By contrast, search::Manager::Params pertains to a single search::Manager instance.
 */
struct SearchParams {
  static SearchParams make_pondering_params(int limit) { return SearchParams{limit, true, true}; }

  int tree_size_limit = 100;
  bool full_search = true;
  bool ponder = false;
};

}  // namespace search
