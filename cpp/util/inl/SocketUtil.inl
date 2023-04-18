#include <util/SocketUtil.hpp>

#include <arpa/inet.h>
#include <netdb.h>
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

inline void Socket::shutdown() {
  ::shutdown(fd_, SHUT_RDWR);
}

inline Socket* Socket::create_server_socket(io::port_t port, int max_connections) {
  auto fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    throw util::Exception("Could not create socket");
  }

  sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = INADDR_ANY;
  if (bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
    throw util::Exception("Could not bind socket");
  }

  if (listen(fd, max_connections) < 0) {
    throw util::Exception("Could not listen on socket");
  }

  return get_instance(fd);
}

inline Socket* Socket::create_client_socket(std::string const& host, port_t port) {
  auto fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    throw util::Exception("Could not create socket");
  }

  struct hostent* entry = gethostbyname(host.c_str());

  sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = inet_addr(inet_ntoa(*(struct in_addr *) *entry->h_addr_list));
  if (connect(fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
    throw util::Exception("Could not connect to socket");
  }

  return get_instance(fd);
}

inline Socket* Socket::accept() const {
  auto fd = ::accept(fd_, nullptr, nullptr);
  if (fd < 0) {
    throw util::Exception("Could not accept connection");
  }

  return get_instance(fd);
}

inline Socket::Reader::Reader(Socket* socket) : socket_(socket) {
  socket->read_mutex_.lock();
}

inline Socket::Reader::~Reader() {
  if (!released_) {
    socket_->read_mutex_.unlock();
  }
}

inline int Socket::Reader::read(char* data, int size) {
  if (released_) {
    throw util::Exception("Socket::Reader::read() called after release()");
  }
  auto n = recv(socket_->fd_, data, size, 0);
  if (n < 0) {
    throw util::Exception("Could not read from socket");
  }
  return n;
}

inline void Socket::Reader::release() {
  if (released_) {
    throw util::Exception("Socket::Reader::release() called twice");
  }
  released_ = true;
  socket_->read_mutex_.unlock();
}

}  // namespace io
