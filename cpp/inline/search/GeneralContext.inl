#include "search/GeneralContext.hpp"

namespace search {

template <search::concepts::SearchSpec SearchSpec>
void GeneralContext<SearchSpec>::RootInfo::clear() {
  state_step = 0;
  node_index = -1;
  active_seat = -1;
  add_noise = false;

  Rules::init_state(state);
  input_encoder.clear();
  input_encoder.update(state);
}

template <search::concepts::SearchSpec SearchSpec>
GeneralContext<SearchSpec>::GeneralContext(const ManagerParams& mparams,
                                           core::mutex_vec_sptr_t node_mutex_pool)
    : manager_params(mparams),
      pondering_search_params(manager_params.pondering_search_params()),
      aux_state(mparams),
      lookup_table(node_mutex_pool) {}

template <search::concepts::SearchSpec SearchSpec>
void GeneralContext<SearchSpec>::clear() {
  aux_state.clear();
  lookup_table.clear();
  root_info.clear();
}

template <search::concepts::SearchSpec SearchSpec>
void GeneralContext<SearchSpec>::step() {
  aux_state.step();
}

template <search::concepts::SearchSpec SearchSpec>
void GeneralContext<SearchSpec>::jump_to(StateIterator it, core::step_t step) {
  root_info.state = it->state;
  root_info.state_step++;
  root_info.input_encoder.jump_to(it);
  aux_state.jump_to(step);
}

}  // namespace search
