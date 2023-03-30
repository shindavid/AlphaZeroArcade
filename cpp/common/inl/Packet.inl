#include <common/Packet.hpp>

#include <sys/socket.h>

#include <boost/lexical_cast.hpp>

#include <util/Exception.hpp>

namespace common {

//template<int MaxPayloadSize>
//void Packet<MaxPayloadSize>::snprintf(const char* format, ...) {
//  va_list args;
//  va_start(args, format);
//  int n = std::vsnprintf(&payload_[header_.payload_size], kMaxPayloadSize - header_.payload_size, format, args);
//  va_end(args);
//
//  if (n < 0) {
//    throw util::Exception("Error formatting payload");
//  }
//
//  header_.payload_size += n;
//  if (header_.payload_size > kMaxPayloadSize) {
//    throw util::Exception("Packet::snprintf() overflow (%d > %d, n=%d)", header_.payload_size, kMaxPayloadSize, n);
//  }
//}
//
//template<int MaxPayloadSize>
//Packet<MaxPayloadSize> Packet<MaxPayloadSize>::parse(const char* ptr) {
//  Packet packet = *reinterpret_cast<const Packet*>(ptr);
//  const PacketHeader& header = packet.header_;
//  if (header.type >= PacketHeader::kNumTypes) {
//    throw util::Exception("Packet::parse() invalid type (%d)", header.type);
//  }
//  if (header.payload_size > kMaxPayloadSize) {
//    throw util::Exception("Packet::parse() overflow (%d > %d)", header.payload_size, kMaxPayloadSize);
//  }
//  return packet;
//}

inline Registration Registration::fromPayloadString(const std::string& s) {
  size_t space = s.find(' ');
  if (space == std::string::npos) {
    throw util::Exception("Registration::fromPayloadString() invalid payload (%s)", s.c_str());
  }
  std::string s0 = s.substr(0, space);
  std::string s1 = s.substr(space + 1);

  Registration registration;
  registration.requested_seat = boost::lexical_cast<int>(s0);
  registration.player_name = s1;

  return registration;
}


inline Registration Packet::to_registration() const {
  if (header.type != PacketHeader::kRegister) {
    throw util::Exception("Packet::to_registration() invalid type (%d)", header.type);
  }
  return Registration::fromPayloadString(payload);
}

inline Packet Packet::from_socket(int socket_descriptor, char* buf, int buf_size) {
  int n = recv(socket_descriptor, buf, buf_size, 0);
  if (n <= 0) {
    throw util::Exception("Packet::from_socket() recv() failed");
  }
  if (n >= buf_size) {
    throw util::Exception("Packet::from_socket() potential buffer overflow (%d >= %d)", n, buf_size);
  }

  PacketHeader header = *reinterpret_cast<const PacketHeader*>(buf);
  if (header.type >= PacketHeader::kNumTypes) {
    throw util::Exception("Packet::parse() invalid type (%d)", header.type);
  }
  Packet packet;
  packet.header = header;
  packet.payload = buf + sizeof(PacketHeader);
  return packet;
}

}  // namespace common
