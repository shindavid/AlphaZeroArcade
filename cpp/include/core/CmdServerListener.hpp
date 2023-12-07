#pragma once

#include <util/Exception.hpp>

namespace core {

/*
 * A connection to a cmd-server can be initiated via core::CmdServerClient::init(). Once that
 * connection is established, any number of CmdServerListeners can subscribe to the connection.
 *
 * Currently, the messages sent from the cmd-server are single-character commands. Each listener
 * is responsible for parsing the command and acting on it.
 */
class CmdServerListener {
 public:
  virtual ~CmdServerListener() = default;

  /*
   * Subscribe to messages from the CmdServerClient.
   */
  void subscribe();

  /*
   * Handle a message from the cmd-server.
   */
  virtual void handle_cmd_server_msg(char msg) = 0;
};

}  // namespace core
