#include <core/LoopControllerClient.hpp>
#include <core/LoopControllerListener.hpp>
#include <core/PerfStats.hpp>
#include <util/Asserts.hpp>
#include <util/CppUtil.hpp>
#include <util/Exception.hpp>
#include <util/LoggingUtil.hpp>

#include <chrono>
#include <thread>
#include <vector>

namespace core {

LoopControllerClient* LoopControllerClient::instance_ = nullptr;

void LoopControllerClient::init(const Params& params) {
  if (instance_) {
    throw util::Exception("LoopControllerClient already initialized");
  }

  instance_ = new LoopControllerClient(params);
}

void LoopControllerClient::handle_worker_ready() {
  LOG_INFO("LoopControllerClient::{}()", __func__);
  std::unique_lock lock(receipt_mutex_);
  worker_ready_count_++;
  if (worker_ready_count_ == (int)worker_ready_listeners_.size()) {
    lock.unlock();
    send_worker_ready();
  }
}

void LoopControllerClient::send_worker_ready() {
  LOG_INFO("LoopControllerClient::{}()", __func__);
  boost::json::object msg;
  msg["type"] = "worker-ready";
  if (params_.weights_request_generation >= 0) {
    msg["gen"] = params_.weights_request_generation;
  }
  send(msg);
}

void LoopControllerClient::handle_pause_receipt() {
  std::unique_lock lock(receipt_mutex_);
  pause_receipt_count_++;

  if (pause_receipt_count_ == pause_listeners_.size()) {
    lock.unlock();
    receipt_cv_.notify_all();
  }
  LOG_INFO("LoopControllerClient::{}() [{} of {}]", __func__,
           pause_receipt_count_, pause_listeners_.size());
}

void LoopControllerClient::handle_unpause_receipt() {
  std::unique_lock lock(receipt_mutex_);
  unpause_receipt_count_++;

  if (unpause_receipt_count_ == pause_listeners_.size()) {
    lock.unlock();
    receipt_cv_.notify_all();
  }
  LOG_INFO("LoopControllerClient: {}() [{} of {}]", __func__,
           unpause_receipt_count_, pause_listeners_.size());
}

void LoopControllerClient::update_perf_stats(PerfStats& stats) {
  std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  perf_stats_.total_time_ns += util::to_ns(now - get_perf_stats_time_);
  get_perf_stats_time_ = now;

  stats.update(perf_stats_);
  perf_stats_ = LoopControllerPerfStats();  // reset
}

LoopControllerClient::LoopControllerClient(const Params& params)
    : PerfStatsClient(), params_(params), proc_start_ts_(util::ns_since_epoch()) {
  if (role().empty()) {
    throw util::CleanException("--client-role must be specified");
  }
  socket_ = io::Socket::create_client_socket(params.loop_controller_hostname,
                                             params.loop_controller_port);
  send_handshake();
  recv_handshake();
}

LoopControllerClient::~LoopControllerClient() { shutdown(); }

void LoopControllerClient::start() {
  thread_ = new std::thread([this]() { loop(); });
}

void LoopControllerClient::shutdown() {
  if (shutdown_initiated_) return;

  shutdown_initiated_ = true;
  send_done();
  socket_->shutdown();

  if (thread_ && thread_->joinable()) {
    thread_->detach();
  }
  delete thread_;
}

void LoopControllerClient::send_done() {
  boost::json::object msg;
  msg["type"] = "done";
  send(msg);
}

void LoopControllerClient::send_with_file(const boost::json::value& msg,
                                          const std::vector<char>& buf) {
  socket_->json_write_and_send_file_bytes(msg, buf);
}

void LoopControllerClient::send_handshake() {
  boost::json::object msg;
  msg["type"] = "handshake";
  msg["role"] = role();
  msg["start_timestamp"] = proc_start_ts_;
  msg["cuda_device"] = cuda_device();
  if (role() == "ratings-worker") {
    boost::json::object aux;
    aux["tag"] = ratings_tag();
    msg["aux"] = aux;
  }
  if (params_.manager_id >= 0) {
    msg["manager_id"] = params_.manager_id;
  }
  socket_->json_write(msg);
}

void LoopControllerClient::recv_handshake() {
  boost::json::value msg;
  if (!socket_->json_read(&msg)) {
    throw util::Exception("{}(): unexpected loop-controller socket close", __func__);
  }

  std::string type = msg.at("type").as_string().c_str();
  util::release_assert(type == "handshake-ack", "Expected handshake-ack, got {}", type);

  if (msg.as_object().contains("rejection")) {
    std::string rejection = msg.at("rejection").as_string().c_str();
    throw util::CleanException("LoopControllerClient handshake rejected: {}", rejection);
  }

  int64_t client_id = msg.at("client_id").as_int64();
  util::release_assert(client_id >= 0, "Invalid client_id {}", client_id);

  client_id_ = client_id;
}

void LoopControllerClient::send_pause_ack() {
  boost::json::object msg;
  msg["type"] = "pause-ack";
  socket_->json_write(msg);
}

void LoopControllerClient::send_unpause_ack() {
  boost::json::object msg;
  msg["type"] = "unpause-ack";
  socket_->json_write(msg);
}

void LoopControllerClient::pause() {
  LOG_INFO("LoopControllerClient: pausing...");
  paused_ = true;
  pause_receipt_count_ = 0;
  pause_time_ = std::chrono::steady_clock::now();

  for (auto listener : pause_listeners_) {
    listener->pause();
  }
}

void LoopControllerClient::unpause() {
  LOG_INFO("LoopControllerClient: unpausing...");
  unpause_receipt_count_ = 0;

  for (auto listener : pause_listeners_) {
    listener->unpause();
  }
}

void LoopControllerClient::reload_weights(const std::vector<char>& buf) {
  LOG_INFO("LoopControllerClient: reloading weights...");

  for (auto listener : reload_weights_listeners_) {
    listener->reload_weights(buf);
  }
}

void LoopControllerClient::handle_data_request(int n_rows, int next_row_limit) {
  LOG_INFO("LoopControllerClient::{}({}, {})...", __func__, n_rows, next_row_limit);

  for (auto listener : data_request_listeners_) {
    listener->handle_data_request(n_rows, next_row_limit);
  }
}

void LoopControllerClient::handle_data_pre_request(int n_rows_limit) {
  LOG_INFO("LoopControllerClient: handling self-play data pre-request({})...", n_rows_limit);

  for (auto listener : data_request_listeners_) {
    listener->handle_data_pre_request(n_rows_limit);
  }
}

void LoopControllerClient::wait_for_pause_receipts() {
  LOG_INFO("LoopControllerClient: waiting for pause receipts...");
  std::unique_lock lock(receipt_mutex_);
  receipt_cv_.wait(lock, [this]() { return pause_receipt_count_ == pause_listeners_.size(); });
  LOG_INFO("LoopControllerClient: pause receipts received!");
}

void LoopControllerClient::wait_for_unpause_receipts() {
  std::unique_lock lock(receipt_mutex_);
  receipt_cv_.wait(lock, [this]() { return unpause_receipt_count_ == pause_listeners_.size(); });
  LOG_INFO("LoopControllerClient: unpause receipts received!");

  if (paused_) {
    auto now = std::chrono::steady_clock::now();
    perf_stats_.pause_time_ns += util::to_ns(now - pause_time_);
    paused_ = false;
  }
}

void LoopControllerClient::loop() {
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

      // reload-weights msg will be immediately followed by a file transfer
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

}  // namespace core
