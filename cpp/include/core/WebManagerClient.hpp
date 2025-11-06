#pragma once

#include "core/AbstractPlayer.hpp"
#include "boost/json/object.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

namespace core {
struct WebManagerClient{
  virtual ~WebManagerClient() = default;
  void handle_start_game() {
    mit::unique_lock lock(mutex_);
    new_game_handled_ = true;
    lock.unlock();
    cv_.notify_all();
  }

  virtual core::seat_index_t seat() const = 0;
  virtual void handle_action(const boost::json::object& payload) = 0;
  virtual void handle_resign() = 0;

 protected:
  void wait_for_new_game() {
    mit::unique_lock lock(mutex_);
    cv_.wait(lock, [this]() { return new_game_handled_; });
    new_game_handled_ = false;
  }

  mit::mutex mutex_;
  mit::condition_variable cv_;
  bool new_game_handled_ = true;
};
}  // namespace core
