#include "search/GeneralContext.hpp"

namespace search {

template <search::concepts::Traits Traits>
void GeneralContext<Traits>::RootInfo::clear() {
  node_index = -1;

  State state;
  Rules::init_state(state);
  input_tensorizor.clear();
  input_tensorizor.update(state);
}

template <search::concepts::Traits Traits>
GeneralContext<Traits>::GeneralContext(const ManagerParams& mparams,
                                       core::mutex_vec_sptr_t node_mutex_pool)
    : manager_params(mparams),
      pondering_search_params(manager_params.pondering_search_params()),
      aux_state(mparams),
      lookup_table(node_mutex_pool) {}

template <search::concepts::Traits Traits>
void GeneralContext<Traits>::clear() {
  aux_state.clear();
  lookup_table.clear();
  root_info.clear();
}

template <search::concepts::Traits Traits>
void GeneralContext<Traits>::step() {
  aux_state.step();
}

template <search::concepts::Traits Traits>
void GeneralContext<Traits>::jump_to(StateIterator it, core::step_t step) {
  root_info.input_tensorizor.jump_to(it);
  aux_state.jump_to(step);
}

}  // namespace search
