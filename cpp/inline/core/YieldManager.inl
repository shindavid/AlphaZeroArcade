#include <core/YieldManager.hpp>

#include <util/LoggingUtil.hpp>

namespace core {

inline void YieldManager::notify(const core::slot_context_vec_t& vec) {
  if (vec.empty()) return;

  std::unique_lock lock(mutex_);
  for (const auto& item : vec) {
    LOG_DEBUG("<-- YieldManager::notify(item={}:{})", item.slot, item.context);
    queue_.push(item);
  }
  pending_queue_count_ -= vec.size();
  lock.unlock();
  cv_.notify_all();
}

inline void YieldManager::notify(const SlotContext& item) {
  LOG_DEBUG("<-- YieldManager::notify(item={}:{})", item.slot, item.context);
  std::unique_lock lock(mutex_);
  queue_.push(item);
  pending_queue_count_--;
  lock.unlock();
  cv_.notify_all();
}

}  // namespace core
