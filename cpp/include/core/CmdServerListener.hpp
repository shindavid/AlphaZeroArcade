#pragma once

#include <util/Exception.hpp>

#include <boost/json.hpp>

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
   * Handle a message from the cmd-server.
   *
   * type is the value of type= in the msg.
   */
  virtual void handle_cmd_server_msg(const boost::json::value& msg, const std::string& type) = 0;
};

}  // namespace core
