#include "core/WebManagerClient.hpp"

namespace core {

void WebManagerClient::handle_start_game() {
  mit::unique_lock lock(mutex_);
  new_game_handled_ = true;
  lock.unlock();
  cv_.notify_all();
}

void WebManagerClient::wait_for_new_game() {
  mit::unique_lock lock(mutex_);
  cv_.wait(lock, [this]() { return new_game_handled_; });
  new_game_handled_ = false;
}

}  // namespace core
