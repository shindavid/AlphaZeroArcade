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

void LoopControllerClient::notify_pause_received(PauseListener* listener) {
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
    : proc_start_ts_(util::ns_since_epoch())
    , cuda_device_(params.cuda_device)
    , role_(params.client_role)
    , reserved_client_id_(params.reserved_client_id) {
  if (role_.empty()) {
    throw util::CleanException("--client-role must be specified");
  }
  socket_ = io::Socket::create_client_socket(params.loop_controller_hostname,
                                             params.loop_controller_port);
  cur_generation_ = params.starting_generation;
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

void LoopControllerClient::send_handshake() {
  boost::json::object msg;
  msg["type"] = "handshake";
  msg["role"] = role_;
  msg["start_timestamp"] = proc_start_ts_;
  msg["cuda_device"] = cuda_device_;
  if (reserved_client_id_ >= 0) {
    msg["reserved_client_id"] = reserved_client_id_;
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

  int64_t client_id = msg.at("client_id").as_int64();
  util::release_assert(client_id >= 0, "Invalid client_id %ld", client_id);

  if (reserved_client_id_ >= 0 && client_id != reserved_client_id_) {
    LOG_WARN << "Reserved client_id " << reserved_client_id_ << " not honored, got " << client_id;
  }

  client_id_ = client_id;
}

void LoopControllerClient::pause() {
  std::unique_lock lock(pause_mutex_);
  if (paused_) return;
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
}

void LoopControllerClient::send_metrics() {
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
  msg["type"] = "pause_ack";
  socket_->json_write(msg);
}

void LoopControllerClient::unpause() {
  for (auto listener : pause_listeners_) {
    listener->unpause();
  }
  paused_ = false;
}

void LoopControllerClient::reload_weights(const std::string& model_filename) {
  for (auto listener : reload_weights_listeners_) {
    listener->reload_weights(model_filename);
  }
}

void LoopControllerClient::loop() {
  // TODO: heartbeat checking to make sure server is still alive
  while (true) {
    boost::json::value msg;
    if (!socket_->json_read(&msg)) {
      if (!shutdown_initiated_) {
        LOG_INFO << "Cmd-server socket closed";
      }
      break;
    }

    std::string type = msg.at("type").as_string().c_str();
    LOG_INFO << "LoopControllerClient handling - " << type;
    if (type == "pause") {
      pause();
      send_pause_ack();
    } else if (type == "unpause") {
      unpause();
    } else if (type == "reload-weights") {
      std::string model_filename = msg.at("model_filename").as_string().c_str();
      cur_generation_ = msg.at("generation").as_int64();
      pause();
      send_metrics();
      reload_weights(model_filename);
      unpause();
    } else if (type == "quit") {
      // TODO: add actual quit logic
      break;
    } else {
      throw util::Exception("Unknown loop-controller message type %s", type.c_str());
    }
    LOG_INFO << "LoopControllerClient " << type << " handling complete";
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
