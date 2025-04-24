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
      .template add_option<"cuda-device">(
          po::value<std::string>(&cuda_device)->default_value(cuda_device),
          "cuda device to register to the loop controller. Usually you need to specify this again "
          "for the MCTS player(s)")
      .template add_hidden_option<"client-role">(
          po::value<std::string>(&client_role)->default_value(client_role),
          "loop controller client role")
      .template add_hidden_option<"ratings-tag">(
          po::value<std::string>(&ratings_tag)->default_value(ratings_tag),
          "ratings tag (only relevant if client_role == ratings-worker)")
      .template add_hidden_option<"output-base-dir">(
          po::value<std::string>(&output_base_dir)->default_value(output_base_dir),
          "output base directory (needed for direct-game-log-write optimization)")
      .template add_hidden_option<"manager-id">(
          po::value<int>(&manager_id)->default_value(manager_id),
          "if specified, indicates the client-id of the manager of this process")
      .template add_hidden_option<"weights-request-generation">(
          po::value<int>(&weights_request_generation)->default_value(weights_request_generation),
          "if specified, requests this specific generation from the loop controller whenever "
          "requesting weights")
      .template add_hidden_flag<"report-metrics", "do-not-report-metrics">(
          &report_metrics, "report metrics to loop-controller periodically",
          "do not report metrics to loop-controller");
}

template <typename T>
void LoopControllerClient::add_listener(T* listener) {
  detail::add_listener(pause_listeners_, listener);
  detail::add_listener(reload_weights_listeners_, listener);
  detail::add_listener(data_request_listeners_, listener);
}

}  // namespace core
