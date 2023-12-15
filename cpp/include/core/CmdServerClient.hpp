#pragma once

#include <util/SocketUtil.hpp>

#include <boost/json.hpp>

#include <string>
#include <thread>
#include <vector>

namespace core {

class CmdServerListener;

/*
 * CmdServerClient is used to communicate with an external cmd-server. It is to be used as a
 * singleton, and can further forward messages to any number of listeners.
 *
 * For now, all messages will be in json format. We can revisit this in the future.
 *
 * Usage:
 *
 * core::CmdServerClient::init(host, port);
 *
 * core::CmdServerClient* client = core::CmdServerClient::get();
 * client->send("abc", 3);
 *
 * To listen to messages from the cmd-server, implement the CmdServerListener interface and
 * subscribe to the client via client->add_listener(listener).
 */
class CmdServerClient {
 public:
  using listener_vec_t = std::vector<CmdServerListener*>;

  static void init(const std::string& host, io::port_t port);
  static bool initialized() { return instance_;  }
  static CmdServerClient* get() { return instance_;  }
  int client_id() const { return client_id_; }

  void send(const boost::json::value& msg) { socket_->json_write(msg); }
  void add_listener(CmdServerListener* listener) { listeners_.push_back(listener); }

 private:
  CmdServerClient(const std::string& host, io::port_t port);
  ~CmdServerClient();
  void send_handshake();
  void recv_handshake();
  void loop();

  static CmdServerClient* instance_;

  const uint64_t proc_start_timestamp_;
  io::Socket* socket_;
  std::thread* thread_;
  listener_vec_t listeners_;
  int client_id_ = -1;  // assigned by cmd-server
};

}  // namespace core
