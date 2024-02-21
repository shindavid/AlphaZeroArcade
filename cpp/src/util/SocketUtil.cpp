#include <util/SocketUtil.hpp>

#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <netdb.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#include <util/Exception.hpp>

namespace io {

Socket::map_t Socket::map_;

Socket* Socket::get_instance(file_descriptor_t fd) {
  auto it = map_.find(fd);
  if (it == map_.end()) {
    auto* instance = new Socket(fd);
    map_[fd] = instance;
    return instance;
  } else {
    return it->second;
  }
}

void Socket::json_write(const boost::json::value& json) {
  std::string json_str = boost::json::serialize(json);
  uint32_t length = htonl(static_cast<uint32_t>(json_str.size()));

  std::unique_lock lock(write_mutex_);
  write_helper(&length, sizeof(length), "Could not json_write length to socket");
  write_helper(json_str.c_str(), json_str.size(), "Could not json_write to socket");
}

bool Socket::json_read(boost::json::value* data) {
  std::unique_lock lock(read_mutex_);

  uint32_t length;
  if (!read_helper(&length, sizeof(length), "Could not json_read length from socket")) {
    return false;
  }
  length = ntohl(length);  // ensure correct byte order

  json_buffer_.resize(length);
  if (!read_helper(json_buffer_.data(), length, "Could not json_read from socket")) {
    return false;
  }
  lock.unlock();

  json_str_.assign(json_buffer_.begin(), json_buffer_.end());
  *data = boost::json::parse(json_str_);
  return true;
}

void Socket::shutdown() {
  if (active_) {
    ::shutdown(fd_, SHUT_RDWR);
    active_ = false;
  }
}

Socket* Socket::create_server_socket(io::port_t port, int max_connections) {
  auto fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    throw util::Exception("Could not create socket");
  }
  const int enable = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0) {
    throw util::Exception("setsockopt(SO_REUSEADDR) failed");
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

Socket* Socket::create_client_socket(std::string const& host, port_t port) {
  auto fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    throw util::CleanException("Could not create socket at %s:%d", host.c_str(), port);
  }

  struct hostent* entry = gethostbyname(host.c_str());

  sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = inet_addr(inet_ntoa(*(struct in_addr*)*entry->h_addr_list));

  int retry_count = 5;
  int sleep_time_ms = 100;
  while (connect(fd, (sockaddr*)&addr, sizeof(addr)) < 0 && retry_count > 0) {
    retry_count--;
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_ms));
    sleep_time_ms *= 2;
  }
  if (retry_count == 0) {
    throw util::CleanException("Could not connect to socket at %s:%d", host.c_str(), port);
  }

  return get_instance(fd);
}

Socket* Socket::accept() const {
  auto fd = ::accept(fd_, nullptr, nullptr);
  if (fd < 0) {
    throw util::Exception("Could not accept connection");
  }

  return get_instance(fd);
}

void Socket::write_helper(const void* data, int size, const char* error_msg) {
  int bytes_sent = 0;
  const char* data_ptr = static_cast<const char*>(data);

  while (bytes_sent < size) {
    int n = send(fd_, data_ptr + bytes_sent, size - bytes_sent, 0);
    if (n < 0) {
      throw util::Exception("%s", error_msg);
    }
    bytes_sent += n;
  }
}

bool Socket::read_helper(void* data, int size, const char* error_msg) {
  int bytes_read = 0;
  char* data_ptr = static_cast<char*>(data);

  while (bytes_read < size) {
    int n = recv(fd_, data_ptr + bytes_read, size - bytes_read, 0);
    if (n < -1) {
      throw util::Exception("%s", error_msg);
    } else if (n <= 0) {
      return false;
    }
    bytes_read += n;
  }
  return true;
}

}  // namespace io
