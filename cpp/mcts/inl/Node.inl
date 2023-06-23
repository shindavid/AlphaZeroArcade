#include <mcts/Node.hpp>

#include <util/ThreadSafePrinter.hpp>

namespace mcts {

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::stable_data_t::stable_data_t(Node* p, core::action_index_t a)
    : parent(p)
      , tensorizor(p->stable_data().tensorizor)
      , state(p->stable_data().state)
      , outcome(state.apply_move(a))
      , action(a)
{
  tensorizor.receive_state_change(state, action);
  aux_init();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::stable_data_t::stable_data_t(
    Node* p, core::action_index_t a, const Tensorizor& t, const GameState& s, const GameOutcome& o)
    : parent(p)
      , tensorizor(t)
      , state(s)
      , outcome(o)
      , action(a)
{
  aux_init();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::stable_data_t::stable_data_t(const stable_data_t& data, bool prune_parent)
{
  *this = data;
  if (prune_parent) parent = nullptr;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::stable_data_t::aux_init() {
  valid_action_mask = state.get_valid_actions();
  current_player = state.get_current_player();
  sym_index = bitset_util::choose_random_on_index(tensorizor.get_symmetry_indices(state));
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::evaluation_data_t::evaluation_data_t(const ActionMask& valid_actions)
    : fully_analyzed_actions(~valid_actions) {}

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
inline Node<GameState, Tensorizor>::Node(Node* parent, core::action_index_t action)
    : stable_data_(parent, action)
      , evaluation_data_(stable_data().valid_action_mask) {}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::Node(
    const Tensorizor& tensorizor, const GameState& state, const GameOutcome& outcome)
    : stable_data_(nullptr, -1, tensorizor, state, outcome)
      , evaluation_data_(stable_data().valid_action_mask) {}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Node<GameState, Tensorizor>::Node(const Node& node, bool prune_parent)
    : stable_data_(node.stable_data_, prune_parent)
      , children_data_(node.children_data_)
      , evaluation_data_(node.evaluation_data_)
      , stats_(node.stats_) {}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline std::string Node<GameState, Tensorizor>::genealogy_str() const {
  const char* delim = kNumGlobalActions < 10 ? "" : ":";
  std::vector<std::string> vec;
  const Node* n = this;
  while (n->parent()) {
    vec.push_back(std::to_string(n->action()));
    n = n->parent();
  }

  std::reverse(vec.begin(), vec.end());
  return util::create_string("[%s]", boost::algorithm::join(vec, delim).c_str());
}

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
inline void Node<GameState, Tensorizor>::adopt_children() {
  // This should only be called in contexts where the search-threads are inactive, so we do not need to worry about
  // thread-safety
  for (child_index_t c = 0; c < stable_data_.num_valid_actions(); ++c) {
    Node* child = get_child(c);
    if (child) child->stable_data_.parent = this;
  }
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline typename Node<GameState, Tensorizor>::PolicyTensor
Node<GameState, Tensorizor>::get_counts() const {
  // This should only be called in contexts where the search-threads are inactive, so we do not need to worry about
  // thread-safety

  core::seat_index_t cp = stable_data().current_player;
  bool forced_win = stats_.forcibly_winning[cp];

  if (kEnableThreadingDebug) {
    std::cout << "get_counts()" << std::endl;
    std::cout << "  cp: " << int(cp) << std::endl;
    std::cout << "  forcibly_winning: " << bitset_util::to_string(stats_.forcibly_winning) << std::endl;
    std::cout << "  forced_win: " << forced_win << std::endl;
  }

  PolicyTensor counts;
  counts.setZero();

  for (child_index_t c = 0; c < stable_data_.num_valid_actions(); ++c) {
    Node* child = get_child(c);
    if (child) {
      if (kEnableThreadingDebug) {
        std::cout << "  child[" << c << "]: " << std::endl;
        std::cout << "    action: " << int(child->action()) << std::endl;
        std::cout << "    forcibly_winning: " << bitset_util::to_string(child->stats().forcibly_winning) << std::endl;
        std::cout << "    count: " << child->stats().count << std::endl;
      }
      if (forced_win) {
        counts.data()[child->action()] = child->stats().forcibly_winning[cp];
      } else {
        counts.data()[child->action()] = child->stats().count;
      }
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
  if (forcibly_losing()) return;
  stats_.value_avg = (stats_.value_avg * stats_.count + outcome) / (stats_.count + 1);
  stats_.count++;
  lock.unlock();

  if (parent()) parent()->backprop(outcome);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::backprop_with_virtual_undo(const ValueArray& value) {
  std::unique_lock<std::mutex> lock(stats_mutex_);
  if (forcibly_losing()) return;
  stats_.value_avg += (value - make_virtual_loss()) / stats_.count;
  stats_.virtual_count--;
  lock.unlock();

  if (parent()) parent()->backprop_with_virtual_undo(value);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::virtual_backprop() {
  std::unique_lock<std::mutex> lock(stats_mutex_);
  if (forcibly_losing()) return;
  auto loss = make_virtual_loss();
  stats_.value_avg = (stats_.value_avg * stats_.count + loss) / (stats_.count + 1);
  stats_.count++;
  stats_.virtual_count++;
  lock.unlock();

  if (parent()) parent()->virtual_backprop();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Node<GameState, Tensorizor>::eliminate(
    int thread_id, player_bitset_t& forcibly_winning, player_bitset_t& forcibly_losing,
    ValueArray& accumulated_value, int& accumulated_count)
{
  core::seat_index_t cp = stable_data().current_player;
  bool winning = forcibly_winning[cp];
  bool losing = forcibly_losing[cp];

  std::unique_lock<std::mutex> lock(stats_mutex_);
  if (eliminated()) return;  // possible if concurrent eliminations due to race-condition

  stats_t prev_stats = stats_;
  stats_.forcibly_winning = forcibly_winning;
  stats_.forcibly_losing = forcibly_losing;

  if (losing) {
    // pretend these visits were never made!
    accumulated_value = stats_.value_avg * stats_.count;
    accumulated_count = stats_.count;
    stats_.zero_out();
  } else {
    stats_.remove(accumulated_value, accumulated_count);
  }

  if (kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread_id);
    printer << "eliminate() " << genealogy_str() << " [cp=" << (int)cp << "]";
    printer.endl();
    printer.printf("  forcibly_winning: %s\n", bitset_util::to_string(forcibly_winning).c_str());
    printer.printf("  forcibly_losing: %s\n", bitset_util::to_string(forcibly_losing).c_str());
    printer.printf("  winning: %d\n", int(winning));
    printer.printf("  losing: %d\n", int(losing));
    printer << "  accumulated_value: " << accumulated_value.transpose();
    printer.endl();
    printer << "  accumulated_count: " << accumulated_count;
    printer.endl();
    printer << "  value_avg: " << prev_stats.value_avg.transpose() << " -> " << stats_.value_avg.transpose();
    printer.endl();
    printer << "  count: " << prev_stats.count << " -> " << stats_.count;
    printer.endl();
  }

  lock.unlock();

  if (parent()) {
    parent()->compute_forced_lines(forcibly_winning, forcibly_losing);
    parent()->eliminate(thread_id, forcibly_winning, forcibly_losing, accumulated_value, accumulated_count);
  }
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void Node<GameState, Tensorizor>::compute_forced_lines(
    player_bitset_t& forcibly_winning, player_bitset_t& forcibly_losing) const
{
  forcibly_winning.reset();
  forcibly_losing.reset();
  core::seat_index_t cp = stable_data().current_player;

  for (child_index_t c = 0; c < stable_data_.num_valid_actions(); ++c) {
    Node *child = get_child(c);
    if (!child) continue;
    const player_bitset_t& fw = child->stats().forcibly_winning;
    if (fw[cp]) {
      forcibly_winning = fw;
      break;
    }
    if (c == 0) {
      forcibly_winning = fw;
    } else {
      forcibly_winning &= fw;
    }
  }

  if (forcibly_winning.any()) {
    forcibly_losing = ~forcibly_winning;
    return;
  }

  for (child_index_t c = 0; c < stable_data_.num_valid_actions(); ++c) {
    Node *child = get_child(c);
    if (!child) {
      forcibly_losing.reset();
      break;
    }
    const player_bitset_t& fl = child->stats().forcibly_losing;
    if (c == 0) {
      forcibly_losing = fl;
    } else {
      forcibly_losing &= fl;
    }
  }
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
void Node<GameState, Tensorizor>::mark_as_fully_analyzed() {
  Node* my_parent = parent();
  if (!my_parent) return;

  std::unique_lock<std::mutex> lock(my_parent->evaluation_data_mutex());
  my_parent->evaluation_data_.fully_analyzed_actions[action()] = true;
  bool full = my_parent->evaluation_data_.fully_analyzed_actions.all();
  lock.unlock();
  if (!full) return;

  my_parent->mark_as_fully_analyzed();
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
