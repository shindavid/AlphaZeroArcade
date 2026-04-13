#include "alpha0/GeneralContext.hpp"

namespace alpha0 {

template <alpha0::concepts::Spec Spec>
void GeneralContext<Spec>::RootInfo::clear() {
  state_step = 0;
  node_index = -1;
  active_seat = -1;
  add_noise = false;

  Rules::init_state(state);
  input_encoder.clear();
  input_encoder.update(state);
}

template <alpha0::concepts::Spec Spec>
GeneralContext<Spec>::GeneralContext(const ManagerParams& mparams,
                                        core::mutex_vec_sptr_t node_mutex_pool)
    : manager_params(mparams),
      pondering_search_params(manager_params.pondering_search_params()),
      aux_state(mparams),
      lookup_table(node_mutex_pool) {}

template <alpha0::concepts::Spec Spec>
void GeneralContext<Spec>::clear() {
  aux_state.clear();
  lookup_table.clear();
  root_info.clear();
}

template <alpha0::concepts::Spec Spec>
void GeneralContext<Spec>::step() {
  aux_state.step();
}

template <alpha0::concepts::Spec Spec>
void GeneralContext<Spec>::jump_to(StateIterator it, core::step_t step) {
  root_info.state = it->state;
  root_info.state_step++;
  root_info.input_encoder.jump_to(it);
  aux_state.jump_to(step);
}

}  // namespace alpha0
