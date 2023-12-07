#include <core/CmdServerClient.hpp>

#include <core/CmdServerListener.hpp>
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

CmdServerClient::CmdServerClient(const std::string& host, io::port_t port) {
  socket_ = io::Socket::create_client_socket(host, port);
  thread_ = new std::thread([this]() { loop(); });
}

CmdServerClient::~CmdServerClient() {
  socket_->shutdown();
  if (thread_ && thread_->joinable()) {
    thread_->detach();
  }
  delete thread_;
}

void CmdServerClient::loop() {
  while (true) {
    io::Socket::Reader reader(socket_);
    char msg[1];
    reader.read(msg, 1);
    for (auto* listener : listeners_) {
      listener->handle_cmd_server_msg(msg[0]);
    }
  }
}

}  // namespace core
