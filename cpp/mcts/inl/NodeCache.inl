#include <mcts/NodeCache.hpp>

namespace mcts {

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NodeCache<GameState, Tensorizor>::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [_, submap] : map_) {
    delete submap;
  }
  map_.clear();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NodeCache<GameState, Tensorizor>::clear_before(move_number_t move_number) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto begin = map_.begin();
  auto end = map_.lower_bound(move_number);
  for (auto it = begin; it != end; ++it) {
    delete it->second;
  }
  map_.erase(begin, end);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename NodeCache<GameState, Tensorizor>::Node_asptr NodeCache<GameState, Tensorizor>::fetch_or_create(
    move_number_t move_number, Node* parent, core::action_t action)
{
  GameState state = parent->stable_data().state;
  auto outcome = state.apply_move(action);

  std::lock_guard<std::mutex> lock(mutex_);
  auto submap_it = map_.find(move_number);
  submap_t* submap;
  if (submap_it == map_.end()) {
    submap = new submap_t();
    map_[move_number] = submap;
  } else {
    submap = submap_it->second;
  }
  auto it = submap->find(state);
  if (it == submap->end()) {
    Tensorizor tensorizor = parent->stable_data().tensorizor;
    tensorizor.receive_state_change(state, action);
    (*submap)[state] = std::make_shared<Node>(tensorizor, state, outcome);  // TODO: use memory pool
    return (*submap)[state];
  }
  return it->second;
}


}  // namespace mcts
