#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

/*
 * Unit of data sent between GameServer and remote players.
 *
 * Changes to this file must be backwards-compatible, since we will test against frozen binaries that were compiled
 * using past versions of this file.
 */

namespace common {

#pragma pack(push, 1)
struct PacketHeader {
  enum Type : uint8_t {
    kRegister = 0,
    kStartGame = 1,
    kStateChange = 2,
    kActionPrompt = 3,
    kAction = 4,
    kNumTypes = 5
  };

  Type type;
  int8_t padding[3] = {};
  int payload_size;

  PacketHeader(Type t=kNumTypes, int ps=0) : type(t), payload_size(ps) {}
};
#pragma pack(pop)

static_assert(sizeof(PacketHeader) == 8);

struct Registration {
  std::string player_name;
  int requested_seat;  // negative = random seat

  std::string toPayloadString() const { return std::to_string(requested_seat) + " " + player_name; }
  static Registration fromPayloadString(const std::string& s);
};

struct StartGamePayload {
  int64_t game_id;
  int num_players;
  int seat_assignment;
};

struct Packet {
  Packet() = default;
  Packet(PacketHeader::Type type, const char* ptr, int payload_size) : header(type, payload_size), payload(ptr) {}

  /*
   * Throws exception if header is not of type kRegister.
   */
  Registration to_registration() const;

  /*
   * Receives bytes from a socket, writes those bytes to buf, and does a reinterpret_cast() of those bytes to a Packet.
   *
   * If buf is not large enough to store the contents of the socket, then an exception is thrown.
   */
  static Packet from_socket(int socket_descriptor, char* buf, int buf_size);

  static void to_socket(int socket_descriptor, PacketHeader::Type type, const char* buf, int buf_size);

  template<typename T>
  static void to_socket(int socket_descriptor, PacketHeader::Type type, const T& t) {
    to_socket(socket_descriptor, type, reinterpret_cast<const char*>(&t), sizeof(T));
  }

  PacketHeader header;
  const char* payload;
};

//#pragma pack(push, 1)
//template<int MaxPayloadSize>
//struct Packet {
//  static constexpr int kMaxPayloadSize = MaxPayloadSize;
//
//  Packet(PacketHeader::Type type) : header_(type, 0) {}
//
//  const PacketHeader& header() const { return header_; }
//  PacketHeader::Type type() const { return header_.type; }
//  int payload_size() const { return header_.payload_size; }
//  int size() const { return sizeof(PacketHeader) + header_.payload_size; }
//  const char* char_ptr() const { return reinterpret_cast<const char*>(this); }
//  const char* payload() const { return payload_; }
//
//  /*
//   * Does snprintf() to this->payload_, appropriately updating this->header_.payload_size.
//   *
//   * Throws exception if snprintf() fails or if the payload overflows.
//   */
//  void snprintf(const char* format, ...) __attribute__((format(printf, 2, 3)));
//
//  /*
//   * reinterpret_cast() with checks.
//   */
//  static Packet parse(const char* ptr);
//
//private:
//  PacketHeader header_;
//  char payload_[kMaxPayloadSize];
//};
//#pragma pack(pop)

}  // namespace common

#include <common/inl/Packet.inl>
