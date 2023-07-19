#include <mcts/Node.hpp>

#include <util/ThreadSafePrinter.hpp>

namespace mcts {

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::stable_data_t::stable_data_t(const Node* parent, core::action_index_t a)
: tensorizor(parent->stable_data().tensorizor)
, state(parent->stable_data().state)
, outcome(state.apply_move(a))
, action(a)
{
  tensorizor.receive_state_change(state, action);
  aux_init();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::stable_data_t::stable_data_t(
    core::action_index_t a, const Tensorizor& t, const GameState& s, const GameOutcome& o)
: tensorizor(t)
, state(s)
, outcome(o)
, action(a)
{
  aux_init();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::stable_data_t::aux_init() {
  valid_action_mask = state.get_valid_actions();
  current_player = state.get_current_player();
  sym_index = bitset_util::choose_random_on_index(tensorizor.get_symmetry_indices(state));
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::stats_t::stats_t() {
  value_avg.setZero();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void Node<GameState, Tensorizor>::stats_t::zero_out()
{
  value_avg.setZero();
  count = 0;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void Node<GameState, Tensorizor>::stats_t::remove(const ValueArray& rm_sum, int rm_count) {
  if (count <= rm_count) {
    zero_out();
  } else {
    value_avg = (value_avg * count - rm_sum) / (count - rm_count);
    count -= rm_count;
  }
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::Node(const Node* parent, core::action_index_t action)
: stable_data_(parent, action) {}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::Node(
    const Tensorizor& tensorizor, const GameState& state, const GameOutcome& outcome)
: stable_data_(-1, tensorizor, state, outcome) {}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::debug_dump() const {
  std::cout << "value[" << stats_.count << "]: " << stats_.value_avg.transpose() << std::endl;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::release(Node* protected_child) {
  // If we got here, the Node and its children (besides protected_child) should not be referenced from anywhere, so it
  // should be safe to delete it without worrying about thread-safety.
  for (child_index_t c = 0; c < stable_data_.num_valid_actions(); ++c) {
    Node* child = get_child(c);
    if (!child) continue;
    if (child != protected_child) child->release();
    clear_child(c);  // not needed currently, but might be needed if we switch to smart-pointers
  }

  delete this;
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

  for (child_index_t c = 0; c < stable_data_.num_valid_actions(); ++c) {
    Node* child = get_child(c);
    if (child) {
      if (kEnableThreadingDebug) {
        std::cout << "  child[" << c << "]: " << std::endl;
        std::cout << "    action: " << int(child->action()) << std::endl;
        std::cout << "    count: " << child->stats().count << std::endl;
      }
      counts.data()[child->action()] = child->stats().count;
    }
  }

  if (kEnableThreadingDebug) {
    std::cout << "  counts:\n" << counts << std::endl;
  }
  return counts;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::backprop(const ValueArray& outcome) {
  std::unique_lock<std::mutex> lock(stats_mutex_);
  stats_.value_avg = (stats_.value_avg * stats_.count + outcome) / (stats_.count + 1);
  stats_.count++;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::backprop_with_virtual_undo(const ValueArray& value) {
  std::unique_lock<std::mutex> lock(stats_mutex_);
  stats_.value_avg += (value - make_virtual_loss()) / stats_.count;
  stats_.virtual_count--;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::virtual_backprop() {
  std::unique_lock<std::mutex> lock(stats_mutex_);
  auto loss = make_virtual_loss();
  stats_.value_avg = (stats_.value_avg * stats_.count + loss) / (stats_.count + 1);
  stats_.count++;
  stats_.virtual_count++;
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
inline Node<GameState, Tensorizor>* Node<GameState, Tensorizor>::init_child(child_index_t c) {
  std::lock_guard guard(children_mutex_);

  Node* child = get_child(c);
  if (child) return child;

  const auto& valid_action_mask = stable_data().valid_action_mask;
  core::action_index_t action = bitset_util::get_nth_on_index(valid_action_mask, c);

  child = new Node(this, action);
  children_data_.set(c, child);
  return child;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
Node<GameState, Tensorizor>*
Node<GameState, Tensorizor>::lookup_child_by_action(core::action_index_t action) const {
  return get_child(bitset_util::count_on_indices_before(stable_data().valid_action_mask, action));
}

}  // namespace mcts
