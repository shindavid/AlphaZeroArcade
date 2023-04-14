#include <util/SocketUtil.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <util/Exception.hpp>

namespace io {

inline Socket* Socket::get_instance(file_descriptor_t fd) {
  auto it = map_.find(fd);
  if (it == map_.end()) {
    auto* instance = new Socket(fd);
    map_[fd] = instance;
    return instance;
  } else {
    return it->second;
  }
}

inline void Socket::write(char const* data, int size) {
  std::lock_guard<std::mutex> lock(write_mutex_);
  auto n = send(fd_, data, size, 0);
  if (n < 0) {
    throw util::Exception("Could not write to socket");
  }
}

inline Socket::Reader::Reader(Socket* socket) : socket_(socket) {
  socket->read_mutex_.lock();
}

inline Socket::Reader::~Reader() {
  if (!released_) {
    socket_->read_mutex_.unlock();
  }
}

inline void Socket::Reader::read(char* data, int size) {
  if (released_) {
    throw util::Exception("Socket::Reader::read() called after release()");
  }
  auto n = recv(socket_->fd_, data, size, 0);
  if (n < 0) {
    throw util::Exception("Could not read from socket");
  }
}

inline void Socket::Reader::release() {
  if (released_) {
    throw util::Exception("Socket::Reader::release() called twice");
  }
  released_ = true;
  socket_->read_mutex_.unlock();
}

}  // namespace io
