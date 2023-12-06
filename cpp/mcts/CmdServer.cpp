#include <mcts/CmdServer.hpp>

namespace mcts {

CmdServerForwarder::forwarder_map_t CmdServerForwarder::map_;

void CmdServerListener::subscribe(const std::string& host, io::port_t port) {
  io::host_port_t host_port{host, port};
  CmdServerForwarder::get(host_port)->add(this);
}

CmdServerForwarder* CmdServerForwarder::get(const io::host_port_t& host_port) {
  auto it = map_.find(host_port);
  if (it == map_.end()) {
    auto* forwarder = new CmdServerForwarder(host_port);
    map_.emplace(host_port, forwarder);
    return forwarder;
  } else {
    return it->second;
  }
}

void CmdServerForwarder::add(CmdServerListener* listener) { listeners_.push_back(listener); }

CmdServerForwarder::CmdServerForwarder(const io::host_port_t& host_port) {
  try {
    socket_ = io::Socket::create_client_socket(host_port.host, host_port.port);
  } catch (...) {
    throw util::CleanException("Could not connect to cmd server at %s:%d", host_port.host.c_str(),
                               host_port.port);
  }
  thread_ = new std::thread([this]() { loop(); });
}

CmdServerForwarder::~CmdServerForwarder() {
  socket_->shutdown();
  if (thread_ && thread_->joinable()) {
    thread_->detach();
  }
  delete thread_;
}

void CmdServerForwarder::loop() {
  while (true) {
    io::Socket::Reader reader(socket_);
    char msg[1];
    reader.read(msg, 1);
    for (auto* listener : listeners_) {
      listener->handle_cmd_server_msg(msg[0]);
    }
  }
}

}  // namespace mcts
