#include <common/DerivedTypes.hpp>

namespace common {

template<typename GameState>
typename GameStateTypes<GameState>::LocalPolicyProbDistr
GameStateTypes<GameState>::global_to_local(const PolicyEigenTensor& policy, const ActionMask& mask) {
  LocalPolicyProbDistr out;
  global_to_local(policy, mask, out);
  return out;
}

template<typename GameState>
void GameStateTypes<GameState>::global_to_local(
    const PolicyEigenTensor& policy, const ActionMask& mask, LocalPolicyProbDistr& out)
{
  out.resize(mask.count());
  int i = 0;
  for (action_index_t action : bitset_util::on_indices(mask)) {
    out[i++] = policy(action);
  }
}

template<typename GameState>
typename GameStateTypes<GameState>::PolicyEigenTensor
GameStateTypes<GameState>::local_to_global(const LocalPolicyProbDistr& policy, const ActionMask& mask) {
  PolicyEigenTensor out;
  local_to_global(policy, mask, out);
  return out;
}

template<typename GameState>
void GameStateTypes<GameState>::local_to_global(
    const LocalPolicyProbDistr& policy, const ActionMask& mask, PolicyEigenTensor& out)
{
  PolicyArray& out_array = eigen_util::reinterpret_as_array(out);
  out_array.setConstant(0);
  int i = 0;
  for (action_index_t action : bitset_util::on_indices(mask)) {
    out_array[action] = policy(i++);
  }
  if (out_array.sum()) {
    out_array /= out_array.sum();
  } else {
    for (action_index_t action : bitset_util::on_indices(mask)) {
      out_array[action] = 1.0f / mask.count();
    }
  }
}

template<typename GameState>
math::var_bindings_map_t GameStateTypes<GameState>::get_var_bindings() {
  math::var_bindings_map_t bindings;
  bindings["b"] = kMaxNumLocalActions;
  return bindings;
}

}  // namespace common
