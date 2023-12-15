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

CmdServerClient::CmdServerClient(const std::string& host, io::port_t port)
    : proc_start_timestamp_(util::ns_since_epoch()) {
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
  msg["proc_start_timestamp"] = proc_start_timestamp_;
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

void CmdServerClient::loop() {
  while (true) {
    boost::json::value msg;
    if (!socket_->json_read(&msg)) {
      throw util::Exception("Unexpected cmd-server socket close");
    }

    std::string type = msg.at("type").as_string().c_str();
    for (auto* listener : listeners_) {
      listener->handle_cmd_server_msg(msg, type);
    }
  }
}

}  // namespace core
