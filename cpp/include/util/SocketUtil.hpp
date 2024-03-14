#pragma once

#include <boost/json.hpp>

#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace io {

using file_descriptor_t = int;
using port_t = int;

struct host_port_t {
  std::string host;
  port_t port;

  auto tie() const { return std::tie(host, port); }
  bool operator<(const host_port_t& other) const { return tie() < other.tie(); }
  bool operator==(const host_port_t& other) const { return tie() == other.tie(); }
};

/*
 * Provides thread-safe access to a socket.
 *
 * The main methods are write() and read(). These methods are thread-safe and loop until all
 * requested bytes are written/read.
 *
 * In some contexts, you may wish to do multiple read() calls in a row. For example, you may wish
 * to read a header and then read the payload. For this usage, we have a Reader class that holds
 * on to the read-mutex for the duration of its lifetime.
 *
 * Example usage:
 *
 * Socket* socket = Socket::create_client_socket(host, port);
 * socket->write("hello", 5);
 *
 * {
 *   Socket::Reader reader(socket);  // acquires read-mutex of socket
 *   reader.read(&header, sizeof(header));
 *   reader.read(&payload, header.size);
 * }  // reader destructor releases read-mutex of socket
 *
 * For convenience, there are json_read() and json_write() methods specialized for json format
 * messages. These messages are prefixed with a 4-byte length header, followed by a serialized
 * json string of that length. (TODO: consider making JsonSocket a derived class of Socket)
 */
class Socket {
 public:
  using map_t = std::map<file_descriptor_t, Socket*>;

  static Socket* get_instance(file_descriptor_t fd);

  /*
   * Thread-safe write to socket. Loops until size bytes are written.
   */
  void write(const void* data, int size);

  /*
   * Thread-safe convenience method for writing json messages. Prepends a 4-byte length header to
   * a serialized json string, and calls write().
   */
  void json_write(const boost::json::value& json);

  /*
   * Thread-safe read from socket.
   *
   * If the socket has been closed, then returns false.
   *
   * Otherwise, loops until size bytes have been read, and returns true.
   *
   * If you need multiple read()'s in a row (like header + payload), consider using the Reader class
   * instead.
   */
  bool read(void* data, int size);

  /*
   * Thread-safe convenience method for reading json messages.
   *
   * If the socket has been closed, then returns false.
   *
   * Otherwise, reads a 4-byte length, and then loops until that many more bytes have been read.
   * Deserializes those bytes into a json object, and returns true.
   */
  bool json_read(boost::json::value* data);

  /*
   * Thread-safe convenience method for reading a file from the socket. Extends the provided
   * stringstream with the file bytes.
   *
   * The file is assumed to have been sent by the send_file() method from the python file
   * py/util/socket_util.py. See that file for data format details.
   *
   * Returns false if the socket has been closed.
   *
   * NOTE: This is inefficient currently in that it copies the file bytes into a vector before
   * copying them into the stringstream. I use a stringstream because that is ultimately what we
   * need to pass to torch::load() (or so I think!), and I didn't see an obvious way to avoid the
   * intermediate vector given this requirement. There is probably a better way to do this.
   */
  bool recv_file_bytes(std::stringstream& ss);

  void shutdown();

  static Socket* create_server_socket(port_t port, int max_connections);
  static Socket* create_client_socket(std::string const& host, port_t port);
  Socket* accept() const;

  /*
   * Helper subclass that facilitates thread-safe reading. Instances of this class hold on to the
   * read-mutex of the socket for the duration of their lifetime.
   */
  class Reader {
   public:
    Reader(Socket* socket);
    ~Reader();

    /*
     * If the socket has been closed, then returns false.
     *
     * Otherwise, loops until all bytes have been read, and returns true.
     */
    bool read(void* data, int size);
    void release();

   private:
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    Socket* socket_;
    bool released_ = false;
  };

 private:
  Socket(file_descriptor_t fd) : fd_(fd) {}
  Socket(const Socket&) = delete;
  Socket& operator=(const Socket&) = delete;

  void write_helper(const void* data, int size, const char* error_msg);
  bool read_helper(void* data, int size, const char* error_msg);

  static map_t map_;

  mutable std::mutex write_mutex_;
  mutable std::mutex read_mutex_;
  const file_descriptor_t fd_;
  std::vector<char> json_buffer_;
  std::string json_str_;
  bool active_ = true;
};

}  // namespace io

#include <inline/util/SocketUtil.inl>
