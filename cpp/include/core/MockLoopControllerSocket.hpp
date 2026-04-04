#pragma once

#ifndef MIT_TEST_MODE
static_assert(false, "MockLoopControllerSocket is only for use in MIT_TEST_MODE");
#endif

#include "util/Exceptions.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <boost/json.hpp>

#include <deque>
#include <string>
#include <vector>

namespace core {

/*
 * A mock socket for use in LoopControllerClientT unit tests.
 *
 * The test "server" side calls push_incoming() to enqueue messages that the client will read via
 * json_read(). The test then reads back what the client sent via pop_outgoing().
 *
 * recv_file_bytes() consumes a pre-queued binary payload pushed via push_incoming_file().
 *
 * All operations are protected by a mit::mutex so they interact correctly with the MIT scheduler.
 *
 * shutdown() sets a flag so that subsequent json_read() calls return false, simulating a closed
 * connection.
 */
class MockLoopControllerSocket {
 public:
  // --- Test-server API ---

  // Enqueue a JSON message for the client to read.
  void push_incoming(const boost::json::value& msg) {
    mit::lock_guard<mit::mutex> lock(mutex_);
    incoming_msgs_.push_back(msg);
  }

  // Enqueue a raw binary payload for the client to read via recv_file_bytes().
  void push_incoming_file(std::vector<char> buf) {
    mit::lock_guard<mit::mutex> lock(mutex_);
    incoming_files_.push_back(std::move(buf));
  }

  // Retrieve the next message the client wrote. Throws if the queue is empty.
  boost::json::value pop_outgoing() {
    mit::lock_guard<mit::mutex> lock(mutex_);
    if (outgoing_msgs_.empty()) {
      throw util::Exception("MockLoopControllerSocket::pop_outgoing(): no messages queued");
    }
    boost::json::value msg = std::move(outgoing_msgs_.front());
    outgoing_msgs_.pop_front();
    return msg;
  }

  int outgoing_size() const {
    mit::lock_guard<mit::mutex> lock(mutex_);
    return (int)outgoing_msgs_.size();
  }

  // --- LoopControllerClientT socket API ---

  void json_write(const boost::json::value& msg) {
    mit::lock_guard<mit::mutex> lock(mutex_);
    outgoing_msgs_.push_back(msg);
  }

  bool json_read(boost::json::value* out) {
    mit::lock_guard<mit::mutex> lock(mutex_);
    if (closed_) return false;
    if (incoming_msgs_.empty()) {
      throw util::Exception("MockLoopControllerSocket::json_read(): no messages queued");
    }
    *out = std::move(incoming_msgs_.front());
    incoming_msgs_.pop_front();
    return true;
  }

  bool recv_file_bytes(std::vector<char>& buf) {
    mit::lock_guard<mit::mutex> lock(mutex_);
    if (closed_) return false;
    if (incoming_files_.empty()) {
      throw util::Exception("MockLoopControllerSocket::recv_file_bytes(): no file queued");
    }
    buf = std::move(incoming_files_.front());
    incoming_files_.pop_front();
    return true;
  }

  void json_write_and_send_file_bytes(const boost::json::value& msg,
                                      const std::vector<char>& /*buf*/) {
    json_write(msg);
  }

  void shutdown() {
    mit::lock_guard<mit::mutex> lock(mutex_);
    closed_ = true;
  }

  // No-op; the mock doesn't need a factory method, but LoopControllerClientT::init() calls
  // Socket::create_client_socket(). Tests use init_for_test() instead, which bypasses this.
  static MockLoopControllerSocket* create_client_socket(const std::string&, int) {
    throw util::Exception("MockLoopControllerSocket: use init_for_test(), not init()");
  }

 private:
  mutable mit::mutex mutex_;
  std::deque<boost::json::value> incoming_msgs_;
  std::deque<std::vector<char>> incoming_files_;
  std::deque<boost::json::value> outgoing_msgs_;
  bool closed_ = false;
};

}  // namespace core
