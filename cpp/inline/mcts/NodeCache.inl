#include <mcts/NodeCache.hpp>

namespace mcts {

template <core::concepts::Game Game>
void NodeCache<Game>::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [_, submap] : map_) {
    delete submap;
  }
  map_.clear();
}

template <core::concepts::Game Game>
void NodeCache<Game>::clear_before(move_number_t move_number) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto begin = map_.begin();
  auto end = map_.lower_bound(move_number);
  for (auto it = begin; it != end; ++it) {
    delete it->second;
  }
  map_.erase(begin, end);
}

template <core::concepts::Game Game>
typename NodeCache<Game>::Node_sptr
NodeCache<Game>::fetch_or_create(move_number_t move_number, const FullState& state,
                                 const ActionOutcome& outcome, const ManagerParams* params) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto submap_it = map_.find(move_number);
  submap_t* submap;
  if (submap_it == map_.end()) {
    submap = new submap_t();
    map_[move_number] = submap;
  } else {
    submap = submap_it->second;
  }
  auto mcts_key = state.mcts_key();
  auto it = submap->find(mcts_key);
  if (it == submap->end()) {
    // TODO: use memory pool
    auto node = std::make_shared<Node>(state, outcome, params);
    (*submap)[mcts_key] = node;
    return node;
  }
  return it->second;
}

}  // namespace mcts
