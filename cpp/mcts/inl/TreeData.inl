#include <mcts/TreeData.hpp>

namespace mcts {

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void TreeData<GameState, Tensorizor>::set_root_softmax_temperature(
    const math::ExponentialDecay& temp) {
  new (&root_softmax_temperature_) math::ExponentialDecay(temp);
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void TreeData<GameState, Tensorizor>::activate_search() {
  std::unique_lock lock(search_mutex_);
  if (search_active_) return;
  search_active_ = true;
  lock.unlock();

  search_begin_cv_.notify_all();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void TreeData<GameState, Tensorizor>::deactivate_search() {
  std::unique_lock lock(search_mutex_);
  if (!search_active_) return;
  search_active_ = false;
  lock.unlock();

  search_end_cv_.notify_all();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void TreeData<GameState, Tensorizor>::increment_active_prefetch_thread_count() {
  std::unique_lock lock(search_mutex_);
  active_prefetch_thread_count_++;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void TreeData<GameState, Tensorizor>::decrement_active_prefetch_thread_count() {
  std::unique_lock lock(search_mutex_);
  active_prefetch_thread_count_--;
  if (active_prefetch_thread_count_ == 0) {
    lock.unlock();
    search_end_cv_.notify_all();
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void TreeData<GameState, Tensorizor>::wait_for_search_activation() {
  std::unique_lock lock(search_mutex_);
  search_begin_cv_.wait(lock, [&]() { return search_active_ || shutdown_initiated_; });
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void TreeData<GameState, Tensorizor>::wait_for_prefetch_threads(bool require_search_inactive) {
  if (require_search_inactive) {
    std::unique_lock lock(search_mutex_);
    search_end_cv_.wait(lock,
                        [&]() { return !search_active_ && active_prefetch_thread_count_ == 0; });
  } else {
    std::unique_lock lock(search_mutex_);
    search_end_cv_.wait(lock, [&]() { return active_prefetch_thread_count_ == 0; });
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void TreeData<GameState, Tensorizor>::shutdown() {
  std::unique_lock lock(search_mutex_);
  shutdown_initiated_ = true;
  lock.unlock();
  search_begin_cv_.notify_all();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
bool TreeData<GameState, Tensorizor>::wait_for_eval(Node* root, Node* node,
                                                    int root_over_prefetch_limit) {
  auto state = node->evaluation_data().state;
  if (state == Node::kSet) return true;

  int root_search_count = root->stats(kSearchMode).real_count;
  int root_prefetch_count_threshold = root_search_count + root_over_prefetch_limit;

  std::unique_lock lock(prefetch_mutex_);
  prefetch_cv_.wait(lock, [&]() {
    state = node->evaluation_data().state;
    return state == Node::kSet ||
           root->stats(kPrefetchMode).real_count >= root_prefetch_count_threshold;
  });

  if (state != Node::kSet) {
    if (mcts::kEnableDebug) {
      util::ThreadSafePrinter printer;
      printer.printf("%s() timeout (prefetch:%d vs search:%d)", __func__,
                     root->stats(kPrefetchMode).real_count, root->stats(kSearchMode).real_count);
      printer << std::endl;
    }
  }
  return state == Node::kSet;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename TreeData<GameState, Tensorizor>::edge_t* TreeData<GameState, Tensorizor>::wait_for_edge(
    Node* root, Node* node, core::action_index_t action_index, int root_over_prefetch_limit) {
  auto& children_data = node->children_data();
  edge_t* edge = children_data.find(action_index);
  if (edge) return edge;

  int root_search_count = root->stats(kSearchMode).real_count;
  int root_prefetch_count_threshold = root_search_count + root_over_prefetch_limit;

  std::unique_lock lock(prefetch_mutex_);
  prefetch_cv_.wait(lock, [&]() {
    edge = children_data.find(action_index);
    return edge || root->stats(kPrefetchMode).real_count >= root_prefetch_count_threshold;
  });

  if (!edge) {
    if (mcts::kEnableDebug) {
      util::ThreadSafePrinter printer;
      printer.printf("%s() timeout (prefetch:%d vs search:%d)", __func__,
                     root->stats(kPrefetchMode).real_count, root->stats(kSearchMode).real_count);
      printer << std::endl;
    }
  }
  return edge;
}

}  // namespace mcts
