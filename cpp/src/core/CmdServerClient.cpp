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

void CmdServerClient::init(const std::string& host, io::port_t port) {
  if (instance_) {
    throw util::Exception("CmdServerClient already initialized");
  }

  instance_ = new CmdServerClient(host, port);
}

bool CmdServerClient::ready_for_pause_ack() {
  for (auto listener : pause_listeners_) {
    if (!listener->ready_for_pause_ack_) {
      return false;
    }
  }
  return true;
}

void CmdServerClient::pause_ack() {
  // all pause listeners have acked
  boost::json::object msg;
  msg["type"] = "pause_ack";
  socket_->json_write(msg);

  for (auto listener : pause_listeners_) {
    listener->ready_for_pause_ack_ = false;
  }
}

bool CmdServerClient::ready_for_flush_games_ack() {
  for (auto listener : flush_games_listeners_) {
    if (!listener->ready_for_flush_games_ack_) {
      return false;
    }
  }
  return true;
}

void CmdServerClient::flush_games_ack() {
  // all flush games listeners have acked
  boost::json::object msg;
  msg["type"] = "flush_games_ack";
  socket_->json_write(msg);

  for (auto listener : flush_games_listeners_) {
    listener->ready_for_flush_games_ack_ = false;
  }
}

perf_stats_t CmdServerClient::get_perf_stats() const {
  perf_stats_t stats;

  for (auto listener : metrics_request_listeners_) {
    stats += listener->get_perf_stats();
  }
  return stats;
}

CmdServerClient::CmdServerClient(const std::string& host, io::port_t port)
    : proc_start_ts_(util::ns_since_epoch()) {
  socket_ = io::Socket::create_client_socket(host, port);
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
  socket_->json_write(msg);
}

void CmdServerClient::recv_handshake() {
  boost::json::value msg;
  if (!socket_->json_read(&msg)) {
    throw util::Exception("Unexpected cmd-server socket close");
  }

  std::string type = msg.at("type").as_string().c_str();
  util::release_assert(type == "handshake_ack", "Expected handshake_ack, got %s", type.c_str());

  int64_t client_id = msg.at("client_id").as_int64();
  util::release_assert(client_id >= 0, "Invalid client_id %ld", client_id);

  client_id_ = client_id;
}

void CmdServerClient::handle_pause() {
  for (auto listener : pause_listeners_) {
    listener->pause();
  }
}

void CmdServerClient::handle_reload_weights(const std::string& model_filename) {
  for (auto listener : reload_weights_listeners_) {
    listener->reload_weights(model_filename);
  }
}

void CmdServerClient::handle_metrics_request() {
  perf_stats_t stats = get_perf_stats();

  int64_t timestamp = util::ns_since_epoch();

  boost::json::object response;
  response["type"] = "metrics_report";
  response["timestamp"] = timestamp;
  response["metrics"] = stats.to_json();
  send(response);

  set_last_metrics_ts(timestamp);
}

void CmdServerClient::handle_flush_games(int next_generation) {
  for (auto listener : flush_games_listeners_) {
    listener->flush_games(next_generation);
  }
}

void CmdServerClient::loop() {
  while (true) {
    boost::json::value msg;
    if (!socket_->json_read(&msg)) {
      throw util::Exception("Unexpected cmd-server socket close");
    }

    std::string type = msg.at("type").as_string().c_str();
    if (type == "pause") {
      handle_pause();
    } else if (type == "reload_weights") {
      std::string model_filename = msg.at("model_filename").as_string().c_str();
      handle_reload_weights(model_filename);
    } else if (type == "metrics_request") {
      handle_metrics_request();
    } else if (type == "flush_games") {
      int next_generation = msg.at("next_generation").as_int64();
      handle_flush_games(next_generation);
    } else {
      throw util::Exception("Unknown cmd-server message type %s", type.c_str());
    }
  }
}

}  // namespace core
