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

void LoopControllerClient::notify_pause_received(PauseListener* listener) {
  LOG_INFO << "LoopControllerClient: received pause notification";
  std::unique_lock lock(pause_mutex_);
  listener->pause_notified_ = true;
  pause_complete_ = all_pause_notifications_received();

  if (pause_complete_) {
    lock.unlock();
    pause_cv_.notify_all();
  }
}

perf_stats_t LoopControllerClient::get_perf_stats() const {
  perf_stats_t stats;

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
  thread_ = new std::thread([this]() { loop(); });
}

LoopControllerClient::~LoopControllerClient() { shutdown(); }

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

void LoopControllerClient::send_with_file(const boost::json::value& msg, std::stringstream& ss) {
  socket_->json_write_and_send_file_bytes(msg, ss);
}

void LoopControllerClient::send_handshake() {
  boost::json::object msg;
  msg["type"] = "handshake";
  msg["role"] = role();
  msg["start_timestamp"] = proc_start_ts_;
  msg["cuda_device"] = cuda_device();
  socket_->json_write(msg);
}

void LoopControllerClient::recv_handshake() {
  boost::json::value msg;
  if (!socket_->json_read(&msg)) {
    throw util::Exception("%s(): unexpected loop-controller socket close", __func__);
  }

  std::string type = msg.at("type").as_string().c_str();
  util::release_assert(type == "handshake-ack", "Expected handshake-ack, got %s", type.c_str());

  int64_t client_id = msg.at("client_id").as_int64();
  util::release_assert(client_id >= 0, "Invalid client_id %ld", client_id);

  client_id_ = client_id;
}

void LoopControllerClient::pause() {
  std::unique_lock lock(pause_mutex_);
  if (paused_) {
    LOG_INFO << "LoopControllerClient: skipping pause (already paused)";
    return;
  }
  LOG_INFO << "LoopControllerClient: pausing...";
  paused_ = true;
  pause_complete_ = pause_listeners_.empty();
  lock.unlock();

  for (auto listener : pause_listeners_) {
    listener->pause_notified_ = false;
  }
  for (auto listener : pause_listeners_) {
    listener->pause();
  }

  lock.lock();
  pause_cv_.wait(lock, [this]() { return pause_complete_; });
  LOG_INFO << "LoopControllerClient: pause complete!";
}

void LoopControllerClient::send_metrics() {
  if (!params_.report_metrics) {
    return;
  }

  LOG_INFO << "LoopControllerClient: sending metrics...";
  int64_t timestamp = util::ns_since_epoch();

  boost::json::object msg;
  msg["type"] = "metrics";
  msg["gen"] = cur_generation_;
  msg["timestamp"] = timestamp;
  msg["metrics"] = get_perf_stats().to_json();

  set_last_games_flush_ts(timestamp);
  send(msg);
}

void LoopControllerClient::send_pause_ack() {
  boost::json::object msg;
  msg["type"] = "pause-ack";
  socket_->json_write(msg);
}

void LoopControllerClient::unpause() {
  LOG_INFO << "LoopControllerClient: unpausing...";
  for (auto listener : pause_listeners_) {
    listener->unpause();
  }
  paused_ = false;
}

void LoopControllerClient::reload_weights(std::stringstream& ss, const std::string& cuda_device) {
  LOG_INFO << "LoopControllerClient: reloading weights...";
  for (auto listener : reload_weights_listeners_) {
    listener->reload_weights(ss, cuda_device);
  }
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
      send_pause_ack();
    } else if (type == "unpause") {
      unpause();
    } else if (type == "reload-weights") {
      std::string cuda_device = this->cuda_device();
      if (msg.as_object().contains("cuda_device")) {
        cuda_device = msg.at("cuda_device").as_string().c_str();
      }
      int64_t generation = msg.at("generation").as_int64();

      // reload-weights msg will be immediately followed by a file transfer
      std::stringstream ss;
      if (!socket_->recv_file_bytes(ss)) {
        if (!shutdown_initiated_) {
          LOG_INFO << "LoopControllerClient: cmd-server socket closed, breaking";
        }
        break;
      }

      pause();
      send_metrics();
      cur_generation_ = generation;
      reload_weights(ss, cuda_device);
      unpause();
    } else if (type == "quit") {
      // TODO: add actual quit logic
      break;
    } else {
      throw util::Exception("Unknown loop-controller message type %s", type.c_str());
    }
    LOG_INFO << "LoopControllerClient: " << type << " handling complete";
  }
}

bool LoopControllerClient::all_pause_notifications_received() const {
  for (auto listener : pause_listeners_) {
    if (!listener->pause_notified_) {
      return false;
    }
  }
  return true;
}

}  // namespace core
