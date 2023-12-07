#pragma once

#include <util/SocketUtil.hpp>

#include <string>
#include <thread>
#include <vector>

namespace core {

class CmdServerListener;

/*
 * An external cmd-server can issue commands to this process. In order to subscribe to these
 * commands, one must invoke CmdServerClient::init() with the host and port of the cmd-server.
 * Then, any number of CmdServerListeners can subscribe to the connection via
 * CmdServerListener::subscribe().
 */
class CmdServerClient {
 public:
  using listener_vec_t = std::vector<CmdServerListener*>;

  static void init(const std::string& host, io::port_t port);
  static bool initialized() { return instance_;  }
  static CmdServerClient* get() { return instance_;  }

  void add(CmdServerListener* listener) { listeners_.push_back(listener); }

 private:
  CmdServerClient(const std::string& host, io::port_t port);
  ~CmdServerClient();
  void loop();

  static CmdServerClient* instance_;
  io::Socket* socket_;
  std::thread* thread_;
  listener_vec_t listeners_;
};

}  // namespace core
