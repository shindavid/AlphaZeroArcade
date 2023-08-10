#include <mcts/Node.hpp>

#include <util/ThreadSafePrinter.hpp>

namespace mcts {

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::stable_data_t::stable_data_t(
    const Tensorizor& t, const GameState& s, const GameOutcome& o)
: tensorizor(t)
, state(s)
, outcome(o)
, valid_action_mask(s.get_valid_actions())
, current_player(s.get_current_player())
, sym_index(bitset_util::choose_random_on_index(tensorizor.get_symmetry_indices(s))) {}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::stats_t::stats_t() {
  eval.setZero();
  real_avg.setZero();
  virtualized_avg.setZero();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::edge_t*
Node<GameState, Tensorizor>::edge_t::instantiate(
    core::action_index_t a, core::local_action_index_t l, asptr c)
{
  const_cast<asptr&>(child_).store(c.load());
  local_action_ = l;
  action_ = a;
  return this;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::edge_t*
Node<GameState, Tensorizor>::edge_chunk_t::find(core::local_action_index_t l)
{
  for (edge_t& edge : data) {
    if (!edge.instantiated()) return nullptr;
    if (edge.local_action() == l) return &edge;
  }
  return next ? next->find(l) : nullptr;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::edge_t*
Node<GameState, Tensorizor>::edge_chunk_t::insert(
    core::action_index_t a, core::local_action_index_t l, asptr child)
{
  for (edge_t& edge : data) {
    if (edge.action() == a) return &edge;
    if (edge.action() == -1) return edge.instantiate(a, l, child);
  }
  if (!next) next = new edge_chunk_t();
  return next->insert(a, l, child);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
template<bool is_const>
Node<GameState, Tensorizor>::children_data_t::template iterator_base_t<is_const>::iterator_base_t(
    chunk_t* chunk, int index)
: chunk(chunk)
, index(index)
{
  nullify_if_at_end();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
template<bool is_const>
void Node<GameState, Tensorizor>::children_data_t::template iterator_base_t<is_const>::increment() {
  index++;
  if (index >= kEdgeDataChunkSize) {
    chunk = chunk->next;
    index = 0;
  }
  nullify_if_at_end();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
template<bool is_const>
void Node<GameState, Tensorizor>::children_data_t::template iterator_base_t<is_const>::nullify_if_at_end() {
  if (chunk && !chunk->data[index].instantiated()) {
    chunk = nullptr;
    index = 0;
  }
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::Node(
    const Tensorizor& tensorizor, const GameState& state, const GameOutcome& outcome)
: stable_data_(tensorizor, state, outcome) {}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::debug_dump() const {
  std::cout << "value[" << stats_.count << "]: " << stats_.value_avg.transpose() << std::endl;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline typename Node<GameState, Tensorizor>::PolicyTensor
Node<GameState, Tensorizor>::get_counts() const {
  // This should only be called in contexts where the search-threads are inactive, so we do not need to worry about
  // thread-safety

  core::seat_index_t cp = stable_data().current_player;

  if (kEnableThreadingDebug) {
    std::cout << "get_counts()" << std::endl;
    std::cout << "  cp: " << int(cp) << std::endl;
  }

  PolicyTensor counts;
  counts.setZero();

  for (auto& it : children_data_) {
    core::action_index_t action = it.action();
    int count = it.child()->stats().real_count;
    if (kEnableThreadingDebug) {
      std::cout << "  " << action << ": " << count << std::endl;
    }
    counts.data()[action] = count;
  }

  return counts;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename Node<GameState, Tensorizor>::ValueArray
Node<GameState, Tensorizor>::make_virtual_loss() const {
  constexpr float x = 1.0 / (kNumPlayers - 1);
  ValueArray virtual_loss;
  virtual_loss.setZero();
  virtual_loss(stable_data().current_player) = x;
  return virtual_loss;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
template<typename UpdateT>
void Node<GameState, Tensorizor>::update_stats(const UpdateT& update_instruction) {
  ValueArray real_sum;
  real_sum.setZero();
  int real_count = 0;
  for (const edge_t& edge : children_data_) {
    int count = edge.count();
    real_sum += edge.child()->stats().real_avg * count;
    real_count += count;
  }

  if (stats_.real_count) {
    real_sum += stats_.eval;
    real_count++;
  }

  ValueArray real_avg = real_count ? (real_sum / real_count) : real_sum;

  std::unique_lock lock(stats_mutex_);
  update_instruction(this);
  stats_.real_avg = real_avg;
  if (stats_.virtual_count) {
    ValueArray virtualized_num = real_sum + make_virtual_loss() * stats_.virtual_count;
    int virtualized_den = real_count + stats_.virtual_count;
    stats_.virtualized_avg = virtualized_num / virtualized_den;
  } else {
    stats_.virtualized_avg = real_avg;
  }
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename Node<GameState, Tensorizor>::asptr
Node<GameState, Tensorizor>::lookup_child_by_action(core::action_index_t action) const {
  for (const edge_t& edge : children_data_) {
    if (edge.action() == action) {
      return edge.child();
    }
  }
  return asptr();
}

}  // namespace mcts
