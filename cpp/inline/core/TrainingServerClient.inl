#include <core/TrainingServerClient.hpp>

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

inline auto TrainingServerClient::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("training-server options");

  return desc
      .template add_option<"training-server-hostname">(
          po::value<std::string>(&training_server_hostname)->default_value(training_server_hostname),
          "training server hotsname")
      .template add_option<"training-server-port">(
          po::value<io::port_t>(&training_server_port)->default_value(training_server_port),
          "training server port. If unset, then this runs without a training server")
      .template add_option<"starting-generation">(
          po::value<int>(&starting_generation)->default_value(starting_generation),
          "starting generation")
      .template add_option<"cuda-device">(
          po::value<std::string>(&cuda_device)->default_value(cuda_device), "cuda device");
}

template <typename T>
void TrainingServerClient::add_listener(T* listener) {
  detail::add_listener(pause_listeners_, listener);
  detail::add_listener(reload_weights_listeners_, listener);
  detail::add_listener(metrics_request_listeners_, listener);
}

}  // namespace core
