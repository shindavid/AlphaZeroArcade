#include <mcts/Node.hpp>

#include <util/CppUtil.hpp>
#include <util/LoggingUtil.hpp>

namespace mcts {

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::stable_data_t::stable_data_t(const GameState& s,
                                                                 const GameOutcome& o,
                                                                 const ManagerParams* mp)
    : outcome(o),
      valid_action_mask(s.get_valid_actions()),
      num_valid_actions(eigen_util::count(valid_action_mask)),
      current_player(s.get_current_player()),
      sym_index(make_sym_index(s, *mp)) {}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::stats_t::stats_t() {
  eval.setZero();
  real_avg.setZero();
  virtualized_avg.setZero();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::edge_t::edge_t() {
  GameStateTypes::nullify_action(const_cast<Action&>(action_));
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::edge_t* Node<GameState, Tensorizor>::edge_t::instantiate(
    const Action& a, core::action_index_t i, sptr c) {
  const_cast<sptr&>(child_) = c;
  const_cast<Action&>(action_) = a;
  action_index_ = i;
  return this;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::edge_t* Node<GameState, Tensorizor>::edge_chunk_t::find(
    core::action_index_t i) {
  for (edge_t& edge : data) {
    if (!edge.instantiated()) return nullptr;
    if (edge.action_index() == i) return &edge;
  }
  return next ? next->find(i) : nullptr;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::edge_t* Node<GameState, Tensorizor>::edge_chunk_t::insert(
    const Action& a, core::action_index_t i, sptr child) {
  for (edge_t& edge : data) {
    if (edge.action() == a) return &edge;
    if (edge.action_index() == -1) return edge.instantiate(a, i, child);
  }
  if (!next) {
    edge_chunk_t* chunk = new edge_chunk_t();
    edge_t* edge = chunk->insert(a, i, child);
    next = chunk;
    return edge;
  } else {
    return next->insert(a, i, child);
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
template <bool is_const>
Node<GameState, Tensorizor>::children_data_t::template iterator_base_t<is_const>::iterator_base_t(
    chunk_t* chunk, int index)
    : chunk(chunk), index(index) {
  nullify_if_at_end();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
template <bool is_const>
void Node<GameState, Tensorizor>::children_data_t::template iterator_base_t<is_const>::increment() {
  index++;
  if (index >= kEdgeDataChunkSize) {
    chunk = chunk->next;
    index = 0;
  }
  nullify_if_at_end();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
template <bool is_const>
void Node<GameState,
          Tensorizor>::children_data_t::template iterator_base_t<is_const>::nullify_if_at_end() {
  if (chunk && !chunk->data[index].instantiated()) {
    chunk = nullptr;
    index = 0;
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::Node(const GameState& state,
                                         const GameOutcome& outcome, const ManagerParams* mp)
    : stable_data_(state, outcome, mp) {}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::debug_dump() const {
  std::cout << "value[" << stats_.count << "]: " << stats_.value_avg.transpose() << std::endl;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline typename Node<GameState, Tensorizor>::PolicyTensor Node<GameState, Tensorizor>::get_counts(
    const ManagerParams& params) const {
  // This should only be called in contexts where the search-threads are inactive, so we do not need
  // to worry about thread-safety

  core::seat_index_t cp = stable_data().current_player;

  if (kEnableDebug) {
    std::cout << "get_counts()" << std::endl;
    std::cout << "  cp: " << int(cp) << std::endl;
  }

  PolicyTensor counts;
  counts.setZero();

  bool provably_winning = stats_.provably_winning[cp];
  bool provably_losing = stats_.provably_losing[cp];

  for (auto& it : children_data_) {
    Action action = it.action();
    const auto& stats = it.child()->stats();
    int count = stats.real_count;

    int modified_count = count;
    const char* detail = "";
    if (params.avoid_proven_losers && !provably_losing && stats.provably_losing[cp]) {
      modified_count = 0;
      detail = " (losing)";
    } else if (params.exploit_proven_winners && provably_winning && !stats.provably_winning[cp]) {
      modified_count = 0;
      detail = " (!winning)";
    } else if (provably_winning) {
      detail = " (winning)";
    }

    if (kEnableDebug) {
      std::cout << "  " << util::std_array_to_string(action, "(", ",", ")") << ": " << count;
      if (modified_count != count) {
        std::cout << " -> " << modified_count;
      }
      std::cout << detail << std::endl;
    }

    if (modified_count) {
      counts(action) = modified_count;
    }
  }

  return counts;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename Node<GameState, Tensorizor>::ValueArray Node<GameState, Tensorizor>::make_virtual_loss()
    const {
  constexpr float x = 1.0 / (kNumPlayers - 1);
  ValueArray virtual_loss;
  virtual_loss.setZero();
  virtual_loss(stable_data().current_player) = x;
  return virtual_loss;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
template <typename UpdateT>
void Node<GameState, Tensorizor>::update_stats(const UpdateT& update_instruction) {
  core::seat_index_t cp = stable_data().current_player;

  ValueArray real_sum;
  real_sum.setZero();
  int real_count = 0;

  /*
   * provably winning/losing calculation
   *
   * TODO: generalize this by computing lower/upper utility in games with unbounded/non-zero-sum
   * utilities.
   */
  bool cp_has_winning_move = false;
  int num_children = 0;

  player_bitset_t all_provably_winning;
  player_bitset_t all_provably_losing;
  all_provably_winning.set();
  all_provably_losing.set();
  for (const edge_t& edge : children_data_) {
    const auto& child_stats = edge.child()->stats();
    int count = edge.count();
    real_sum += child_stats.real_avg * count;
    real_count += count;

    cp_has_winning_move |= child_stats.provably_winning[cp];
    all_provably_winning &= child_stats.provably_winning;
    all_provably_losing &= child_stats.provably_losing;
    num_children++;
  }

  std::unique_lock lock(stats_mutex_);
  update_instruction(this);

  if (stats_.real_count) {
    real_sum += stats_.eval;
    real_count++;
  }

  // incorporate bounds from children
  int num_valid_actions = stable_data_.num_valid_actions;
  if (num_valid_actions == 0) {
    // terminal state, provably_winning/losing are already set by instruction
  } else if (cp_has_winning_move) {
    stats_.provably_winning[cp] = true;
    stats_.provably_losing.set();
    stats_.provably_losing[cp] = false;
  } else if (num_children == num_valid_actions) {
    stats_.provably_winning = all_provably_winning;
    stats_.provably_losing = all_provably_losing;
  }

  stats_.real_avg = real_count ? (real_sum / real_count) : real_sum;
  if (stats_.virtual_count) {
    ValueArray virtualized_num = real_sum + make_virtual_loss() * stats_.virtual_count;
    int virtualized_den = real_count + stats_.virtual_count;
    stats_.virtualized_avg = virtualized_num / virtualized_den;
  } else {
    stats_.virtualized_avg = stats_.real_avg;
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename Node<GameState, Tensorizor>::sptr Node<GameState, Tensorizor>::lookup_child_by_action(
    const Action& action) const {
  for (const edge_t& edge : children_data_) {
    if (edge.action() == action) {
      return edge.child();
    }
  }
  return nullptr;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
core::symmetry_index_t Node<GameState, Tensorizor>::make_sym_index(const GameState& state,
                                                                   const ManagerParams& params) {
  if (params.apply_random_symmetries) {
    return bitset_util::choose_random_on_index(state.get_symmetry_indices());
  }
  return 0;
}

}  // namespace mcts
