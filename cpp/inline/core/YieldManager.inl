#include <core/YieldManager.hpp>

#include <util/LoggingUtil.hpp>

namespace core {

inline YieldManager::~YieldManager() {
  shut_down();
}

inline void YieldManager::run(func_t f) {
  thread_ = std::thread([f = std::move(f), this]() {
    loop(f);
  });
}

inline void YieldManager::shut_down() {
  mutex_.lock();
  shutting_down_ = true;
  mutex_.unlock();
  cv_.notify_all();

  if (thread_.joinable()) thread_.join();
}

inline void YieldManager::notify(const core::slot_context_vec_t& vec) {
  if (vec.empty()) return;

  std::unique_lock lock(mutex_);
  for (const auto& item : vec) {
    LOG_DEBUG("<-- YieldManager::notify(item={}:{})", item.slot, item.context);
    ready_items_.push_back(item);
  }
  lock.unlock();
  cv_.notify_all();
}

inline void YieldManager::notify(const SlotContext& item) {
  LOG_DEBUG("<-- YieldManager::notify(item={}:{})", item.slot, item.context);
  std::unique_lock lock(mutex_);
  ready_items_.push_back(item);
  lock.unlock();
  cv_.notify_all();
}


inline void YieldManager::loop(func_t f) {
  while (!shutting_down_) {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this]() { return !ready_items_.empty() || shutting_down_; });
    if (shutting_down_) {
      break;
    }
    f(ready_items_);
    ready_items_.clear();
  }
}

}  // namespace core
