#include <core/LoopControllerClient.hpp>

#include <core/LoopControllerListener.hpp>
#include <util/Asserts.hpp>
#include <util/CppUtil.hpp>
#include <util/Exception.hpp>
#include <util/LoggingUtil.hpp>

#include <map>
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

void LoopControllerClient::request_weights() {
  boost::json::object msg;
  msg["type"] = "weights-request";
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
  LOG_INFO << "LoopControllerClient: handle_pause_receipt() - done (pause_receipt_count_ = "
           << pause_receipt_count_ << ")";
}

void LoopControllerClient::handle_unpause_receipt() {
  std::unique_lock lock(receipt_mutex_);
  unpause_receipt_count_++;

  if (unpause_receipt_count_ == pause_listeners_.size()) {
    lock.unlock();
    receipt_cv_.notify_all();
  }
}

PerfStats LoopControllerClient::get_perf_stats() const {
  PerfStats stats;

  for (auto listener : metrics_request_listeners_) {
    stats += listener->get_perf_stats();
  }
  return stats;
}

LoopControllerClient::LoopControllerClient(const Params& params)
    : params_(params)
    , proc_start_ts_(util::ns_since_epoch()) {
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
    throw util::Exception("%s(): unexpected loop-controller socket close", __func__);
  }

  std::string type = msg.at("type").as_string().c_str();
  util::release_assert(type == "handshake-ack", "Expected handshake-ack, got %s", type.c_str());

  if (msg.as_object().contains("rejection")) {
    std::string rejection = msg.at("rejection").as_string().c_str();
    throw util::CleanException("LoopControllerClient handshake rejected: %s", rejection.c_str());
  }

  int64_t client_id = msg.at("client_id").as_int64();
  util::release_assert(client_id >= 0, "Invalid client_id %ld", client_id);

  client_id_ = client_id;
}

void LoopControllerClient::send_metrics() {
  if (!params_.report_metrics) {
    return;
  }

  PerfStats stats = get_perf_stats();
  if (stats.empty()) {
    return;
  }

  LOG_INFO << "LoopControllerClient: sending metrics...";
  int64_t timestamp = util::ns_since_epoch();

  boost::json::object msg;
  msg["type"] = "metrics";
  msg["gen"] = cur_generation_;
  msg["timestamp"] = timestamp;
  msg["metrics"] = stats.to_json();

  set_last_games_flush_ts(timestamp);
  send(msg);
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
  LOG_INFO << "LoopControllerClient: pausing...";
  pause_receipt_count_ = 0;

  for (auto listener : pause_listeners_) {
    listener->pause();
  }
}

void LoopControllerClient::unpause() {
  LOG_INFO << "LoopControllerClient: unpausing...";
  unpause_receipt_count_ = 0;

  for (auto listener : pause_listeners_) {
    listener->unpause();
  }
}

void LoopControllerClient::reload_weights(const std::vector<char>& buf,
                                          const std::string& cuda_device) {
  LOG_INFO << "LoopControllerClient: reloading weights...";

  for (auto listener : reload_weights_listeners_) {
    listener->reload_weights(buf, cuda_device);
  }
}

void LoopControllerClient::wait_for_pause_receipts() {
  LOG_INFO << "LoopControllerClient: waiting for pause receipts...";
  std::unique_lock lock(receipt_mutex_);
  receipt_cv_.wait(lock, [this]() { return pause_receipt_count_ == pause_listeners_.size(); });
  LOG_INFO << "LoopControllerClient: pause receipts received!";
}

void LoopControllerClient::wait_for_unpause_receipts() {
  std::unique_lock lock(receipt_mutex_);
  receipt_cv_.wait(lock, [this]() { return unpause_receipt_count_ == pause_listeners_.size(); });
  LOG_INFO << "LoopControllerClient: unpause receipts received!";
}

void LoopControllerClient::loop() {
  while (true) {
    boost::json::value msg;
    if (!socket_->json_read(&msg)) {
      if (!shutdown_initiated_) {
        LOG_INFO << "LoopControllerClient: cmd-server socket closed, breaking";
      }
      break;
    }

    std::string type = msg.at("type").as_string().c_str();
    LOG_INFO << "LoopControllerClient: handling - " << msg;
    if (type == "pause") {
      pause();
      wait_for_pause_receipts();
      send_pause_ack();
    } else if (type == "unpause") {
      unpause();
      wait_for_unpause_receipts();
      send_unpause_ack();
    } else if (type == "reload-weights") {
      std::string cuda_device = this->cuda_device();
      if (msg.as_object().contains("cuda_device")) {
        cuda_device = msg.at("cuda_device").as_string().c_str();
      }
      int64_t generation = msg.at("generation").as_int64();

      // reload-weights msg will be immediately followed by a file transfer
      std::vector<char> buf;
      if (!socket_->recv_file_bytes(buf)) {
        if (!shutdown_initiated_) {
          LOG_INFO << "LoopControllerClient: cmd-server socket closed, breaking";
        }
        break;
      }

      send_metrics();
      cur_generation_ = generation;
      reload_weights(buf, cuda_device);
    } else if (type == "quit") {
      deactivated_ = true;
      break;
    } else {
      throw util::Exception("Unknown loop-controller message type %s", type.c_str());
    }
    LOG_INFO << "LoopControllerClient: " << type << " handling complete";
  }
}

}  // namespace core
