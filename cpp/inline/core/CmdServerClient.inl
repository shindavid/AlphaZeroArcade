#include <core/CmdServerClient.hpp>

#include <util/BoostUtil.hpp>

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

inline auto CmdServerClient::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("cmd-server options");

  return desc
      .template add_option<"cmd-server-hostname">(
          po::value<std::string>(&cmd_server_hostname)->default_value(cmd_server_hostname),
          "cmd server hotsname")
      .template add_option<"cmd-server-port">(
          po::value<io::port_t>(&cmd_server_port)->default_value(cmd_server_port),
          "cmd server port. If unset, then this runs without a cmd server")
      .template add_option<"starting-generation">(
          po::value<int>(&starting_generation)->default_value(starting_generation),
          "starting generation");
}

template <typename T>
void CmdServerClient::add_listener(T* listener) {
  detail::add_listener(pause_listeners_, listener);
  detail::add_listener(reload_weights_listeners_, listener);
  detail::add_listener(metrics_request_listeners_, listener);
}

}  // namespace core