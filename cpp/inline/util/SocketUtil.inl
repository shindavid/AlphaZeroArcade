#include "util/SocketUtil.hpp"

#include "util/Exceptions.hpp"
#include "util/mit/mit.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>

#include <cstring>
#include <netdb.h>
#include <unistd.h>

namespace io {

inline void Socket::write(const void* data, int size) {
  mit::unique_lock lock(write_mutex_);
  write_helper(data, size, "Could not write to socket");
}

inline bool Socket::read(void* data, int size) {
  mit::unique_lock lock(read_mutex_);
  return read_helper(data, size, "Could not read from socket");
}

inline Socket::Reader::Reader(Socket* socket) : socket_(socket) { socket->read_mutex_.lock(); }

inline Socket::Reader::~Reader() {
  if (!released_) {
    socket_->read_mutex_.unlock();
  }
}

inline bool Socket::Reader::read(void* data, int size) {
  if (released_) {
    throw util::Exception("Socket::Reader::read() called after release()");
  }
  return socket_->read_helper(data, size, "Could not read from socket");
}

inline void Socket::Reader::release() {
  if (released_) {
    throw util::Exception("Socket::Reader::release() called twice");
  }
  released_ = true;
  socket_->read_mutex_.unlock();
}

}  // namespace io
