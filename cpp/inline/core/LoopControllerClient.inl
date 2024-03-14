#include <core/LoopControllerClient.hpp>

#include <util/BoostUtil.hpp>
#include <util/LoggingUtil.hpp>

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

inline auto LoopControllerClient::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("loop-controller options");

  return desc
      .template add_option<"loop-controller-hostname">(
          po::value<std::string>(&loop_controller_hostname)
              ->default_value(loop_controller_hostname),
          "loop controller hotsname")
      .template add_option<"loop-controller-port">(
          po::value<io::port_t>(&loop_controller_port)->default_value(loop_controller_port),
          "loop controller port. If unset, then this runs without a loop controller")
      .template add_option<"client-role">(
          po::value<std::string>(&client_role)->default_value(client_role),
          "loop controller client role")
      .template add_option<"cuda-device">(
          po::value<std::string>(&cuda_device)->default_value(cuda_device),
          "cuda device to register to the loop controller. Usually you need to specify this again "
          "for the MCTS player(s)")
      .template add_option<"weights-request-generation">(
          po::value<int>(&weights_request_generation)->default_value(weights_request_generation),
          "if specified, requests this specific generation from the loop controller whenever "
          "requesting weights");
}

template <typename T>
void LoopControllerClient::add_listener(T* listener) {
  detail::add_listener(pause_listeners_, listener);
  detail::add_listener(reload_weights_listeners_, listener);
  detail::add_listener(metrics_request_listeners_, listener);
}

}  // namespace core
