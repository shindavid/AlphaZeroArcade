#include <core/DerivedTypes.hpp>

namespace core {

template <typename GameState>
bool GameStateTypes<GameState>::is_terminal_outcome(const GameOutcome& outcome) {
  return outcome.sum() > 0;
}

template <typename GameState>
typename GameStateTypes<GameState>::GameOutcome
GameStateTypes<GameState>::make_non_terminal_outcome() {
  GameOutcome outcome;
  outcome.setZero();
  return outcome;
}

template <typename GameState>
typename GameStateTypes<GameState>::LocalPolicyArray GameStateTypes<GameState>::global_to_local(
    const PolicyTensor& policy, const ActionMask& mask) {
  LocalPolicyArray out;
  global_to_local(policy, mask, out);
  return out;
}

template <typename GameState>
void GameStateTypes<GameState>::global_to_local(const PolicyTensor& policy, const ActionMask& mask,
                                                LocalPolicyArray& out) {
  out.resize(eigen_util::count(mask));
  const auto& policy_array = eigen_util::reinterpret_as_array(policy);
  const auto& mask_array = eigen_util::reinterpret_as_array(mask);

  int i = 0;
  for (int a = 0; a < kNumGlobalActionsBound; ++a) {
    if (!mask_array(a)) continue;

    out(i++) = policy_array(a);
    if (i > kMaxNumLocalActions) {
      throw util::Exception("kMaxNumLocalActions too small (%d < %d)", kMaxNumLocalActions, i);
    }
  }
}

template <typename GameState>
typename GameStateTypes<GameState>::PolicyTensor GameStateTypes<GameState>::local_to_global(
    const LocalPolicyArray& policy, const ActionMask& mask) {
  PolicyTensor out;
  local_to_global(policy, mask, out);
  return out;
}

template <typename GameState>
void GameStateTypes<GameState>::local_to_global(const LocalPolicyArray& policy,
                                                const ActionMask& mask, PolicyTensor& out) {
  auto& out_array = eigen_util::reinterpret_as_array(out);
  out_array.setConstant(0);

  const auto& mask_array = eigen_util::reinterpret_as_array(mask);

  int i = 0;
  for (int a = 0; a < kNumGlobalActionsBound; ++a) {
    if (!mask_array(a)) continue;

    out_array(a) = policy(i++);
    if (i > kMaxNumLocalActions) {
      throw util::Exception("kMaxNumLocalActions too small (%d < %d)", kMaxNumLocalActions, i);
    }
  }
}

template <typename GameState>
typename GameStateTypes<GameState>::Action GameStateTypes<GameState>::get_nth_valid_action(
    const ActionMask& valid_actions, int n) {
  const bool* data = valid_actions.data();
  int i = 0;
  for (; i < kNumGlobalActionsBound; ++i) {
    if (!data[i]) continue;
    if (n == 0) break;
    n--;
  }

  return eigen_util::unflatten_index(valid_actions, i);
}

template <typename GameState>
void GameStateTypes<GameState>::nullify_action(Action& action) {
  action.fill(-1);
}

template <typename GameState>
bool GameStateTypes<GameState>::is_nullified(const Action& action) {
  return action[0] == -1;
}

template <typename GameState>
bool GameStateTypes<GameState>::is_valid_action(const Action& action) {
  PolicyShape shape;  // dummy instances to access dimension, constructor should be no-op
  for (size_t i = 0; i < action.size(); ++i) {
    int a = action[i];
    int max_a = shape[i];
    if (a < 0 || a >= max_a) {
      return false;
    }
  }
  return true;
}

template <typename GameState>
bool GameStateTypes<GameState>::is_valid_action(const Action& action,
                                                const ActionMask& valid_actions) {
  return is_valid_action(action) && valid_actions(action);
}

template <typename GameState>
void GameStateTypes<GameState>::validate_action(const Action& action) {
  if (!is_valid_action(action)) {
    throw util::Exception("Invalid action (action=%s)",
                          util::std_array_to_string(action, "(", ",", ")").c_str());
  }
}

template <typename GameState>
void GameStateTypes<GameState>::validate_action(const Action& action,
                                                const ActionMask& valid_actions) {
  if (!is_valid_action(action, valid_actions)) {
    throw util::Exception("Invalid action (action=%s)",
                          util::std_array_to_string(action, "(", ",", ")").c_str());
  }
}

template <typename GameState>
math::var_bindings_map_t GameStateTypes<GameState>::get_var_bindings() {
  math::var_bindings_map_t bindings;
  bindings["b"] = kMaxNumLocalActions;
  return bindings;
}

}  // namespace core
