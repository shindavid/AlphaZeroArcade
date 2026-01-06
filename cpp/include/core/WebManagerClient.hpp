#pragma once

#include "boost/json/object.hpp"
#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <type_traits>

namespace core {
struct WebManagerClient {
  template <class WebManager>
  explicit WebManagerClient(std::in_place_type_t<WebManager> = {}) {
    WebManager::get_instance()->register_client(this);
  }
  virtual ~WebManagerClient() = default;

  void handle_start_game();
  virtual void handle_action(const boost::json::object& payload, seat_index_t seat) = 0;
  virtual void handle_resign(seat_index_t seat) = 0;
  virtual void handle_backtrack(game_tree_index_t index, seat_index_t seat) {}

 protected:
  void wait_for_new_game();

 private:
  mit::mutex mutex_;
  mit::condition_variable cv_;
  bool new_game_handled_ = true;
};

}  // namespace core

#include "inline/core/WebManagerClient.inl"
