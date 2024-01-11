#include <core/CmdServerClient.hpp>

#include <core/CmdServerListener.hpp>
#include <util/Asserts.hpp>
#include <util/CppUtil.hpp>
#include <util/Exception.hpp>

#include <map>
#include <thread>
#include <vector>

namespace core {

CmdServerClient* CmdServerClient::instance_ = nullptr;

void CmdServerClient::init(const Params& params) {
  if (instance_) {
    throw util::Exception("CmdServerClient already initialized");
  }

  instance_ = new CmdServerClient(params);
}

void CmdServerClient::notify_pause_received(PauseListener* listener) {
  std::unique_lock lock(pause_mutex_);
  listener->pause_notified_ = true;
  pause_complete_ = all_pause_notifications_received();

  if (pause_complete_) {
    lock.unlock();
    pause_cv_.notify_all();
  }
}

perf_stats_t CmdServerClient::get_perf_stats() const {
  perf_stats_t stats;

  for (auto listener : metrics_request_listeners_) {
    stats += listener->get_perf_stats();
  }
  return stats;
}

CmdServerClient::CmdServerClient(const Params& params)
    : proc_start_ts_(util::ns_since_epoch()), shared_gpu_(params.shared_gpu) {
  socket_ = io::Socket::create_client_socket(params.cmd_server_hostname, params.cmd_server_port);
  cur_generation_ = params.starting_generation;
  send_handshake();
  recv_handshake();
  thread_ = new std::thread([this]() { loop(); });
}

CmdServerClient::~CmdServerClient() {
  socket_->shutdown();
  if (thread_ && thread_->joinable()) {
    thread_->detach();
  }
  delete thread_;
}

void CmdServerClient::send_handshake() {
  boost::json::object msg;
  msg["type"] = "handshake";
  msg["proc_start_timestamp"] = proc_start_ts_;
  msg["shared_gpu"] = shared_gpu_;
  socket_->json_write(msg);
}

void CmdServerClient::recv_handshake() {
  boost::json::value msg;
  if (!socket_->json_read(&msg)) {
    throw util::Exception("%s(): unexpected cmd-server socket close", __func__);
  }

  std::string type = msg.at("type").as_string().c_str();
  util::release_assert(type == "handshake_ack", "Expected handshake_ack, got %s", type.c_str());

  int64_t client_id = msg.at("client_id").as_int64();
  util::release_assert(client_id >= 0, "Invalid client_id %ld", client_id);

  client_id_ = client_id;
}

void CmdServerClient::pause() {
  pause_complete_ = false;

  for (auto listener : pause_listeners_) {
    listener->pause_notified_ = false;
    listener->pause();
  }

  std::unique_lock lock(pause_mutex_);
  pause_cv_.wait(lock, [this]() { return pause_complete_; });
}

void CmdServerClient::send_metrics() {
  int64_t timestamp = util::ns_since_epoch();

  boost::json::object msg;
  msg["type"] = "metrics";
  msg["gen"] = cur_generation_;
  msg["timestamp"] = timestamp;
  msg["metrics"] = get_perf_stats().to_json();

  set_last_games_flush_ts(timestamp);
  send(msg);
}

void CmdServerClient::send_pause_ack() {
  boost::json::object msg;
  msg["type"] = "pause_ack";
  socket_->json_write(msg);
}

void CmdServerClient::unpause() {
  for (auto listener : pause_listeners_) {
    listener->unpause();
  }
}

void CmdServerClient::reload_weights(const std::string& model_filename) {
  for (auto listener : reload_weights_listeners_) {
    listener->reload_weights(model_filename);
  }
}

void CmdServerClient::loop() {
  while (true) {
    boost::json::value msg;
    if (!socket_->json_read(&msg)) {
      throw util::Exception("%s() unexpected cmd-server socket close", __func__);
    }

    std::string type = msg.at("type").as_string().c_str();
    if (type == "pause") {
      pause();
      send_pause_ack();
    } else if (type == "reload_weights") {
      std::string model_filename = msg.at("model_filename").as_string().c_str();
      cur_generation_ = msg.at("generation").as_int64();
      pause();
      send_metrics();
      reload_weights(model_filename);
      unpause();
    } else {
      throw util::Exception("Unknown cmd-server message type %s", type.c_str());
    }
  }
}

bool CmdServerClient::all_pause_notifications_received() const {
  for (auto listener : pause_listeners_) {
    if (!listener->pause_notified_) {
      return false;
    }
  }
  return true;
}

}  // namespace core
