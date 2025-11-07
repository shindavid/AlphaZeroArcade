#pragma once

#include "core/AbstractPlayer.hpp"
#include "boost/json/object.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <type_traits>

namespace core {
struct WebManagerClient{
  template <class WebManagerT>
  explicit WebManagerClient(std::in_place_type_t<WebManagerT> = {}) {
    WebManagerT::get_instance()->register_client(this);
  }
  virtual ~WebManagerClient() = default;

  void handle_start_game();
  virtual core::seat_index_t seat() const = 0;
  virtual void handle_action(const boost::json::object& payload) = 0;
  virtual void handle_resign() = 0;

 protected:
  void wait_for_new_game();

 private:
  mit::mutex mutex_;
  mit::condition_variable cv_;
  bool new_game_handled_ = true;
};

}  // namespace core

#include "inline/core/WebManagerClient.inl"
