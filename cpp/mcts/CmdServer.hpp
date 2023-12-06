#pragma once

/*
 * Various mcts objects subscribe to cmds issued by a parent cmd-server. This module provides the
 * machinery for this communication.
 *
 * Current cmd set:
 *
 * 'p': pause search threads
 * 'u': unpause search threads
 * 'r': refresh network weights from disk
 */

#include <util/SocketUtil.hpp>

#include <map>
#include <string>
#include <thread>
#include <vector>

namespace mcts {

class CmdServerListener {
 public:
  virtual ~CmdServerListener() = default;
  virtual void handle_cmd_server_msg(char msg) = 0;
  void subscribe(const std::string& host, io::port_t port);
};

class CmdServerForwarder {
 public:
  using forwarder_map_t = std::map<io::host_port_t, CmdServerForwarder*>;
  using listener_vec_t = std::vector<CmdServerListener*>;

  static CmdServerForwarder* get(const io::host_port_t& host_port);
  void add(CmdServerListener* listener);

 private:
  CmdServerForwarder(const io::host_port_t& host_port);
  ~CmdServerForwarder();
  void loop();

  static forwarder_map_t map_;
  io::Socket* socket_;
  std::thread* thread_;
  listener_vec_t listeners_;
};

}  // namespace mcts
