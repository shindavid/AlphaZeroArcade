#pragma once

#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace io {

using file_descriptor_t = int;
using port_t = int;

/*
 * Provides thread-safe access to a socket.
 *
 * Thread-safe writing is simple: just call write() and the mutex will be acquired and released for
 * you.
 *
 * For reading, you may wish to do double-reads, e.g. read a header and then read the payload. For
 * this reason, the interface is slightly more complicated.
 *
 * Socket* socket = Socket::get_instance(fd);
 * socket->write("hello", 5);  // thread-safe, acquires and release write-mutex
 *
 * {
 *   Socket::Reader reader(socket);  // acquires read-mutex of socket
 *   reader.read(buf, 4);  // thread-safe
 * }  // reader destructor releases read-mutex of socket
 *
 * If you wish, you can directly release the read-mutex by calling reader.release(), instead of
 * waiting for the destructor.
 */
class Socket {
 public:
  using map_t = std::map<file_descriptor_t, Socket*>;

  static Socket* get_instance(file_descriptor_t fd);
  void write(char const* data, int size);
  void shutdown();

  static Socket* create_server_socket(port_t port, int max_connections);
  static Socket* create_client_socket(std::string const& host, port_t port);
  Socket* accept() const;

  class Reader {
   public:
    Reader(Socket* socket);
    ~Reader();

    int read(char* data, int size);  // return number of bytes read. 0 means orderly-shutdown
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

  static map_t map_;

  mutable std::mutex write_mutex_;
  mutable std::mutex read_mutex_;
  const file_descriptor_t fd_;
  bool active_ = true;
};

}  // namespace io

#include <util/inl/SocketUtil.inl>
