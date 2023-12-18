#include <core/CmdServerClient.hpp>

#include <concepts>

namespace core {

namespace detail {

template <typename ListenerType, typename T>
void add_listener(std::vector<ListenerType*>& listeners, T* listener) {}

template <typename ListenerType, std::derived_from<ListenerType> T>
void add_listener(std::vector<ListenerType*>& listeners, T* listener) {
  for (auto existing_listener : listeners) {
    if (existing_listener == listener) return;
  }
  listeners.push_back(listener);
}

}  // namespace detail

template <typename T>
void CmdServerClient::add_listener(T* listener) {
  detail::add_listener(pause_listeners_, listener);
  detail::add_listener(reload_weights_listeners_, listener);
  detail::add_listener(metrics_request_listeners_, listener);
  detail::add_listener(flush_games_listeners_, listener);
}

}  // namespace core
