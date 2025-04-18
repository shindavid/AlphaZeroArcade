#include <core/HibernationManager.hpp>

namespace core {

inline HibernationManager::~HibernationManager() {
  shut_down();
}

inline void HibernationManager::run(func_t f) {
  thread_ = std::thread([f = std::move(f), this]() {
    loop(f);
  });
}

inline void HibernationManager::shut_down() {
  mutex_.lock();
  shutting_down_ = true;
  mutex_.unlock();
  cv_.notify_all();

  if (thread_.joinable()) thread_.join();
}

inline void HibernationManager::notify(game_slot_index_t slot_id) {
  std::unique_lock lock(mutex_);
  ready_slots_.push_back(slot_id);
  cv_.notify_all();
}

inline void HibernationManager::loop(func_t f) {
  while (!shutting_down_) {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this]() { return !ready_slots_.empty() || shutting_down_; });
    if (shutting_down_) {
      break;
    }
    game_slot_index_t slot_id = ready_slots_.back();
    f(slot_id);
    ready_slots_.pop_back();
  }
}

}  // namespace core
