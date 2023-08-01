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
  value_avg.setZero();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::stats_t::add(const ValueArray& value) {
  value_avg = (value_avg * count + value) / (count + 1);
  count++;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::stats_t::add_virtual_loss(const ValueArray& loss) {
  value_avg = (value_avg * count + loss) / (count + 1);
  count++;
  virtual_count++;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::stats_t::correct_virtual_loss(const ValueArray& correction) {
  value_avg += correction / count;
  virtual_count--;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
Node<GameState, Tensorizor>::ValueArray Node<GameState, Tensorizor>::stats_t::compute_clipped_update_value(
    const stats_t& edge_stats, float eps) const
{
  ValueArray value_delta = this->value_avg - edge_stats.value_avg;
  if (value_delta.abs().maxCoeff() < eps) {
    ValueArray zero_delta;
    zero_delta.setZero();
    return zero_delta;
  }

  value_delta *= edge_stats.count;
  value_delta += this->value_avg;

  if (abs(value_delta.sum() - 1) > 1e-3) {
    std::ostringstream ss;
    ss << __func__ << " - unexpected value_delta" << std::endl;
    ss << "  this->value_avg: " << this->value_avg.transpose() << std::endl;
    ss << "  edge_stats.value_avg: " << edge_stats.value_avg.transpose() << std::endl;
    ss << "  value_delta: " << value_delta.transpose() << std::endl;
    ss << "  value_delta.sum(): " << value_delta.sum() << std::endl;
    throw util::Exception("%s", ss.str().c_str());
  }

  // clip operation - TODO: change this if we ever remove the sum(value)==1.0 && value>=0 constraint.
  ValueArray clipped_delta = value_delta;
  for (int i = 0; i < kNumPlayers; ++i) {
    if (clipped_delta(i) >= 0) continue;
    dtype pos_sum = eigen_util::positive_sum(clipped_delta);
    dtype factor = (pos_sum + clipped_delta(i)) / pos_sum;
    clipped_delta(i) = 0;
    eigen_util::positive_scale(clipped_delta, factor);
  }

  if (clipped_delta.minCoeff() < 0 || clipped_delta.maxCoeff() > 1.0 + 1e-3 || abs(clipped_delta.sum() - 1) > 1e-3) {
    std::ostringstream ss;
    ss << __func__ << " - unexpected clipped_delta" << std::endl;
    ss << "  value_delta: " << value_delta.transpose() << std::endl;
    ss << "  clipped_delta: " << clipped_delta.transpose() << std::endl;
    ss << "  clipped_delta.sum(): " << clipped_delta.sum() << std::endl;
    ss << "  clipped_delta.minCoeff() < 0: " << (clipped_delta.minCoeff() < 0) << std::endl;
    ss << "  clipped_delta.maxCoeff() > 1.0 + 1e-6: " << (clipped_delta.maxCoeff() > 1.0 + 1e-3) << std::endl;
    ss << "  abs(clipped_delta.sum() - 1) > 1e-3: " << (abs(clipped_delta.sum() - 1) > 1e-3) << std::endl;
    throw util::Exception("%s", ss.str().c_str());
  }

  return clipped_delta;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::edge_data_t*
Node<GameState, Tensorizor>::edge_data_t::instantiate(
    core::action_index_t a, core::local_action_index_t l, asptr c)
{
  const_cast<asptr&>(child_).store(c.load());
  local_action_ = l;
  action_ = a;
  return this;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::edge_data_t*
Node<GameState, Tensorizor>::edge_data_chunk_t::find(core::local_action_index_t l)
{
  for (edge_data_t& edge_data : data) {
    if (!edge_data.instantiated()) return nullptr;
    if (edge_data.local_action() == l) return &edge_data;
  }
  return next ? next->find(l) : nullptr;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::edge_data_t*
Node<GameState, Tensorizor>::edge_data_chunk_t::insert(
    core::action_index_t a, core::local_action_index_t l, asptr child)
{
  for (edge_data_t& edge_data : data) {
    if (edge_data.action() == a) return &edge_data;
    if (edge_data.action() == -1) return edge_data.instantiate(a, l, child);
  }
  if (!next) next = new edge_data_chunk_t();
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
    int count = it.child()->stats().count;
    if (kEnableThreadingDebug) {
      std::cout << "  " << action << ": " << count << std::endl;
    }
    counts.data()[action] = count;
  }

  return counts;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::backprop(ValueArray& value, Node* parent,edge_data_t* edge_data) {
  if (parent) {
    std::unique_lock parent_lock(parent->children_mutex_);

    std::unique_lock child_lock(stats_mutex_);
    bool transposition = edge_data->stats().count != stats_.count;
    if (transposition) {
      value = stats_.compute_clipped_update_value(edge_data->stats(), 0.0);
    }
    edge_data->stats().add(value);
    stats_.add(value);
    return;
  }

  std::unique_lock child_lock(stats_mutex_);
  stats_.add(value);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::backprop_with_virtual_undo(
    ValueArray& value, Node* parent, edge_data_t* edge_data)
{
  if (parent) {
    std::unique_lock parent_lock(parent->children_mutex_);

    std::unique_lock child_lock(stats_mutex_);
    bool transposition = edge_data->stats().count != stats_.count;
    if (transposition) {
      value = stats_.compute_clipped_update_value(edge_data->stats(), 0.0);
    }
    ValueArray correction = value - make_virtual_loss();
    edge_data->stats().correct_virtual_loss(correction);
    stats_.correct_virtual_loss(correction);
    return;
  }

  std::unique_lock child_lock(stats_mutex_);
  stats_.correct_virtual_loss(value - make_virtual_loss());
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::virtual_backprop(Node* parent, edge_data_t* edge_data) {
  if (parent) {
    std::unique_lock parent_lock(parent->children_mutex_);

    ValueArray loss = make_virtual_loss();
    std::unique_lock child_lock(stats_mutex_);
    edge_data->stats().add_virtual_loss(loss);
    stats_.add_virtual_loss(loss);
    return;
  }

  std::unique_lock child_lock(stats_mutex_);
  stats_.add_virtual_loss(make_virtual_loss());
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
typename Node<GameState, Tensorizor>::asptr
Node<GameState, Tensorizor>::lookup_child_by_action(core::action_index_t action) const {
  for (const edge_data_t& edge_data : children_data_) {
    if (edge_data.action() == action) {
      return edge_data.child();
    }
  }
  return asptr();
}

}  // namespace mcts
