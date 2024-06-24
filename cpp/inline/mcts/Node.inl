#include <mcts/Node.hpp>

#include <util/CppUtil.hpp>
#include <util/LoggingUtil.hpp>

namespace mcts {

template <core::concepts::Game Game>
inline Node<Game>::stable_data_t::stable_data_t(const FullState& s,
                                                const ActionOutcome& o,
                                                const ManagerParams* mp)
    : outcome(o),
      valid_action_mask(Rules::get_legal_moves(s)),
      num_valid_actions(valid_action_mask.count()),
      current_player(Rules::get_current_player(s)),
      sym_index(make_sym_index(s, *mp)) {}

template <core::concepts::Game Game>
inline Node<Game>::stats_t::stats_t() {
  eval.setZero();
  real_avg.setZero();
  virtualized_avg.setZero();
}

template <core::concepts::Game Game>
inline Node<Game>::edge_t* Node<Game>::edge_t::instantiate(
    core::action_t a, core::action_index_t i, sptr c) {
  const_cast<sptr&>(child_) = c;
  action_ = a;
  action_index_ = i;
  return this;
}

template <core::concepts::Game Game>
inline Node<Game>::edge_t* Node<Game>::edge_chunk_t::find(core::action_index_t i) {
  for (edge_t& edge : data) {
    if (!edge.instantiated()) return nullptr;
    if (edge.action_index() == i) return &edge;
  }
  return next ? next->find(i) : nullptr;
}

template <core::concepts::Game Game>
inline Node<Game>::edge_t* Node<Game>::edge_chunk_t::insert(
    core::action_t a, core::action_index_t i, sptr child) {
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

template <core::concepts::Game Game>
template <bool is_const>
Node<Game>::children_data_t::template iterator_base_t<is_const>::iterator_base_t(
    chunk_t* chunk, int index)
    : chunk(chunk), index(index) {
  nullify_if_at_end();
}

template <core::concepts::Game Game>
template <bool is_const>
void Node<Game>::children_data_t::template iterator_base_t<is_const>::increment() {
  index++;
  if (index >= kEdgeDataChunkSize) {
    chunk = chunk->next;
    index = 0;
  }
  nullify_if_at_end();
}

template <core::concepts::Game Game>
template <bool is_const>
void Node<Game>::children_data_t::template iterator_base_t<is_const>::nullify_if_at_end() {
  if (chunk && !chunk->data[index].instantiated()) {
    chunk = nullptr;
    index = 0;
  }
}

template <core::concepts::Game Game>
Node<Game>::children_data_t::~children_data_t() {
  edge_chunk_t* next = first_chunk_.next;
  while (next) {
    edge_chunk_t* tmp = next->next;
    delete next;
    next = tmp;
  }
}

template <core::concepts::Game Game>
Node<Game>::Node(const FullState& state, const ActionOutcome& outcome, const ManagerParams* mp)
    : stable_data_(state, outcome, mp) {}

template <core::concepts::Game Game>
void Node<Game>::debug_dump() const {
  std::cout << "value[" << stats_.count << "]: " << stats_.value_avg.transpose() << std::endl;
}

template <core::concepts::Game Game>
typename Node<Game>::PolicyTensor Node<Game>::get_counts(const ManagerParams& params) const {
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
    core::action_t action = it.action();
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
      std::cout << "  " << action << ": " << count;
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

template <core::concepts::Game Game>
typename Node<Game>::ValueArray Node<Game>::make_virtual_loss() const {
  constexpr float x = 1.0 / (kNumPlayers - 1);
  ValueArray virtual_loss;
  virtual_loss.setZero();
  virtual_loss(stable_data().current_player) = x;
  return virtual_loss;
}

template <core::concepts::Game Game>
template <typename UpdateT>
void Node<Game>::update_stats(const UpdateT& update_instruction) {
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

template <core::concepts::Game Game>
typename Node<Game>::sptr Node<Game>::lookup_child_by_action(core::action_t action) const {
  for (const edge_t& edge : children_data_) {
    if (edge.action() == action) {
      return edge.child();
    }
  }
  return nullptr;
}

template <core::concepts::Game Game>
core::symmetry_index_t Node<Game>::make_sym_index(const FullState& state,
                                                  const ManagerParams& params) {
  if (params.apply_random_symmetries) {
    return bitset_util::choose_random_on_index(Rules::get_symmetries(state));
  }
  return 0;
}

}  // namespace mcts
