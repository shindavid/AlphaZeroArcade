#include "core/LoopControllerClient.hpp"

#include "core/PerfStats.hpp"
#include "util/Asserts.hpp"
#include "util/BoostUtil.hpp"
#include "util/CppUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/LoggingUtil.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <chrono>
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

template <typename Socket>
auto LoopControllerClientImpl<Socket>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("loop-controller options");

  return desc
    .template add_option<"loop-controller-hostname">(
      po::value<std::string>(&loop_controller_hostname)->default_value(loop_controller_hostname),
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
      "ratings tag (only relevant if client_role == ratings-worker, eval-vs-benchmark-worker)")
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

template <typename Socket>
void LoopControllerClientImpl<Socket>::init(const Params& params) {
  if (instance_) {
    throw util::Exception("LoopControllerClient already initialized");
  }
  Socket* socket =
    Socket::create_client_socket(params.loop_controller_hostname, params.loop_controller_port);
  instance_ = new LoopControllerClientImpl(params, socket);
}

template <typename Socket>
LoopControllerClientImpl<Socket>::LoopControllerClientImpl(const Params& params, Socket* socket)
    : PerfStatsClient(), params_(params), proc_start_ts_(util::ns_since_epoch()), socket_(socket) {
  if (role().empty()) {
    throw util::CleanException("--client-role must be specified");
  }
  send_handshake();
  recv_handshake();
}

template <typename Socket>
LoopControllerClientImpl<Socket>::~LoopControllerClientImpl() {
  shutdown();
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::start() {
  thread_ = new mit::thread([this]() { loop(); });
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::shutdown() {
  if (shutdown_initiated_) return;
  shutdown_initiated_ = true;
  send_done();
  socket_->shutdown();

  if (thread_ && thread_->joinable()) {
    thread_->detach();
  }
  delete thread_;
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::send_done() {
  boost::json::object msg;
  msg["type"] = "done";
  send(msg);
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::send_with_file(const boost::json::value& msg,
                                                      const std::vector<char>& buf) {
  socket_->json_write_and_send_file_bytes(msg, buf);
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::handle_worker_ready() {
  LOG_INFO("LoopControllerClient::{}()", __func__);
  mit::unique_lock lock(receipt_mutex_);
  worker_ready_count_++;
  if (worker_ready_count_ == (int)worker_ready_listeners_.size()) {
    lock.unlock();
    send_worker_ready();
  }
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::handle_pause_receipt(const char* file, int line) {
  mit::unique_lock lock(receipt_mutex_);
  pause_receipt_count_++;
  if (pause_receipt_count_ == pause_listeners_.size()) {
    lock.unlock();
    receipt_cv_.notify_all();
  }
  LOG_INFO("LoopControllerClient::{}() [{}@{}] [{} of {}]", __func__, file, line,
           pause_receipt_count_, pause_listeners_.size());
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::handle_unpause_receipt(const char* file, int line) {
  mit::unique_lock lock(receipt_mutex_);
  unpause_receipt_count_++;
  if (unpause_receipt_count_ == pause_listeners_.size()) {
    lock.unlock();
    receipt_cv_.notify_all();
  }
  LOG_INFO("LoopControllerClient: {}() [{}@{}] [{} of {}]", __func__, file, line,
           unpause_receipt_count_, pause_listeners_.size());
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::update_perf_stats(PerfStats& stats) {
  std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  perf_stats_.total_time_ns += util::to_ns(now - get_perf_stats_time_);
  get_perf_stats_time_ = now;

  stats.update(perf_stats_);
  perf_stats_ = LoopControllerPerfStats();
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::send_worker_ready() {
  LOG_INFO("LoopControllerClient::{}()", __func__);
  boost::json::object msg;
  msg["type"] = "worker-ready";
  if (params_.weights_request_generation >= 0) {
    msg["gen"] = params_.weights_request_generation;
  }
  send(msg);
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::send_handshake() {
  boost::json::object msg;
  msg["type"] = "handshake";
  msg["role"] = role();
  msg["start_timestamp"] = proc_start_ts_;
  msg["cuda_device"] = cuda_device();

  if (role() == "ratings-worker" || role() == "eval-vs-benchmark-worker") {
    msg["rating_tag"] = ratings_tag();
  }
  if (params_.manager_id >= 0) {
    msg["manager_id"] = params_.manager_id;
  }
  socket_->json_write(msg);
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::recv_handshake() {
  boost::json::value msg;
  if (!socket_->json_read(&msg)) {
    throw util::Exception("{}(): unexpected loop-controller socket close", __func__);
  }

  std::string type = msg.at("type").as_string().c_str();
  RELEASE_ASSERT(type == "handshake-ack", "Expected handshake-ack, got {}", type);

  if (msg.as_object().contains("rejection")) {
    std::string rejection = msg.at("rejection").as_string().c_str();
    throw util::CleanException("LoopControllerClient handshake rejected: {}", rejection);
  }

  int64_t client_id = msg.at("client_id").as_int64();
  RELEASE_ASSERT(client_id >= 0, "Invalid client_id {}", client_id);

  client_id_ = client_id;
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::send_pause_ack() {
  boost::json::object msg;
  msg["type"] = "pause-ack";
  socket_->json_write(msg);
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::send_unpause_ack() {
  boost::json::object msg;
  msg["type"] = "unpause-ack";
  socket_->json_write(msg);
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::pause() {
  LOG_INFO("LoopControllerClient: pausing...");
  paused_ = true;
  pause_receipt_count_ = 0;
  pause_time_ = std::chrono::steady_clock::now();

  for (auto listener : pause_listeners_) {
    listener->pause();
  }
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::unpause() {
  LOG_INFO("LoopControllerClient: unpausing...");
  unpause_receipt_count_ = 0;

  for (auto listener : pause_listeners_) {
    listener->unpause();
  }
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::reload_weights(const std::vector<char>& buf) {
  LOG_INFO("LoopControllerClient: reloading weights...");

  for (auto listener : reload_weights_listeners_) {
    listener->reload_weights(buf);
  }
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::handle_data_request(int n_rows, int next_row_limit) {
  LOG_INFO("LoopControllerClient::{}({}, {})...", __func__, n_rows, next_row_limit);

  for (auto listener : data_request_listeners_) {
    listener->handle_data_request(n_rows, next_row_limit);
  }
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::handle_data_pre_request(int n_rows_limit) {
  LOG_INFO("LoopControllerClient: handling self-play data pre-request({})...", n_rows_limit);

  for (auto listener : data_request_listeners_) {
    listener->handle_data_pre_request(n_rows_limit);
  }
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::wait_for_pause_receipts() {
  LOG_INFO("LoopControllerClient: waiting for pause receipts...");
  mit::unique_lock lock(receipt_mutex_);
  receipt_cv_.wait(lock, [this]() { return pause_receipt_count_ == pause_listeners_.size(); });
  LOG_INFO("LoopControllerClient: pause receipts received!");
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::wait_for_unpause_receipts() {
  mit::unique_lock lock(receipt_mutex_);
  receipt_cv_.wait(lock, [this]() { return unpause_receipt_count_ == pause_listeners_.size(); });
  LOG_INFO("LoopControllerClient: unpause receipts received!");

  if (paused_) {
    auto now = std::chrono::steady_clock::now();
    perf_stats_.pause_time_ns += util::to_ns(now - pause_time_);
    paused_ = false;
  }
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::loop() {
  get_perf_stats_time_ = std::chrono::steady_clock::now();
  while (true) {
    boost::json::value msg;
    if (!socket_->json_read(&msg)) {
      if (!shutdown_initiated_) {
        LOG_INFO("LoopControllerClient: cmd-server socket closed, breaking");
      }
      break;
    }

    std::string type = msg.at("type").as_string().c_str();
    LOG_INFO("LoopControllerClient: handling - {}", fmt::streamed(msg));
    if (type == "pause") {
      pause();
      wait_for_pause_receipts();
      send_pause_ack();
    } else if (type == "unpause") {
      unpause();
      wait_for_unpause_receipts();
      send_unpause_ack();
    } else if (type == "data-request") {
      int n_rows = msg.at("n_rows").as_int64();
      int next_row_limit = msg.at("next_row_limit").as_int64();
      handle_data_request(n_rows, next_row_limit);
    } else if (type == "data-pre-request") {
      int n_rows_limit = msg.at("n_rows_limit").as_int64();
      handle_data_pre_request(n_rows_limit);
    } else if (type == "reload-weights") {
      core::PerfClocker clocker(perf_stats_.model_load_time_ns);

      std::vector<char> buf;
      if (!socket_->recv_file_bytes(buf)) {
        if (!shutdown_initiated_) {
          LOG_INFO("LoopControllerClient: cmd-server socket closed, breaking");
        }
        break;
      }

      reload_weights(buf);
    } else if (type == "quit") {
      deactivated_ = true;
      break;
    } else {
      throw util::Exception("Unknown loop-controller message type {}", type);
    }
    LOG_INFO("LoopControllerClient: {} handling complete", type);
  }
}

template <typename Socket>
template <typename T>
void LoopControllerClientImpl<Socket>::add_listener(T* listener) {
  detail::add_listener(pause_listeners_, listener);
  detail::add_listener(reload_weights_listeners_, listener);
  detail::add_listener(data_request_listeners_, listener);
  detail::add_listener(worker_ready_listeners_, listener);
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::init_for_test(const Params& params, Socket* socket) {
  if (instance_) {
    throw util::Exception("LoopControllerClient already initialized");
  }
  instance_ = new LoopControllerClientImpl(params, socket);
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::reset_for_test() {
  if (instance_) {
    instance_->shutdown_initiated_ = true;  // prevent shutdown() from touching the socket
    delete instance_;
    instance_ = nullptr;
  }
  core::PerfStatsRegistry::clear();
}

template <typename Socket>
void LoopControllerClientImpl<Socket>::join_loop_thread() {
  if (thread_ && thread_->joinable()) {
    thread_->join();
  }
}

}  // namespace core
