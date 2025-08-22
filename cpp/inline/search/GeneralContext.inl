#include "search/GeneralContext.hpp"

namespace search {

template <typename Traits>
void GeneralContext<Traits>::RootInfo::clear() {
  node_index = -1;

  for (group::element_t sym = 0; sym < SymmetryGroup::kOrder; ++sym) {
    history_array[sym].initialize(Rules{});
    State& state = history_array[sym].current();
    Symmetries::apply(state, sym);
  }

  const State& raw_state = history_array[group::kIdentity].current();
  canonical_sym = Symmetries::get_canonical_symmetry(raw_state);
}

template <typename Traits>
GeneralContext<Traits>::GeneralContext(const ManagerParams& mparams,
                                       search::mutex_vec_sptr_t node_mutex_pool)
    : manager_params(mparams),
      pondering_search_params(manager_params.pondering_search_params()),
      lookup_table(node_mutex_pool) {}

template <typename Traits>
void GeneralContext<Traits>::clear() {
  lookup_table.clear();
  root_info.clear();
}

}  // namespace search
