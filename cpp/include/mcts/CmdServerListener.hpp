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

#include <string>

namespace mcts {

class CmdServerListener {
 public:
  virtual ~CmdServerListener() = default;
  virtual void handle_cmd_server_msg(char msg) = 0;
  void subscribe(const std::string& host, io::port_t port);
};

}  // namespace mcts
