#pragma once

#include <array>
#include <concepts>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <common/BasicTypes.hpp>
#include <common/Constants.hpp>

#include <util/CppUtil.hpp>
#include <util/MetaProgramming.hpp>

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
    kRegistration = 0,
    kRegistrationResponse = 1,
    kGameThreadInitialization = 2,
    kGameThreadInitializationResponse = 3,
    kStartGame = 4,
    kStateChange = 5,
    kActionPrompt = 6,
    kAction = 7,
    kNumTypes = 8
  };

  Type type;
  int8_t padding[3] = {};
  int payload_size;

  PacketHeader(Type t=kNumTypes, int ps=0) : type(t), payload_size(ps) {}
};
#pragma pack(pop)

static_assert(sizeof(PacketHeader) == 8);

template <class PayloadType>
concept PacketPayloadConcept = requires(PayloadType p) {
  { util::decay_copy(PayloadType::kType) } -> std::same_as<PacketHeader::Type>;
};

struct Registration {
  static constexpr PacketHeader::Type kType = PacketHeader::kRegistration;
  struct dynamic_size_section_t {
    char player_name[kMaxNameLength + 1];  // +1 for null terminator
  };

  player_index_t requested_seat;  // negative = random seat
  dynamic_size_section_t dynamic_size_section;
};

struct RegistrationResponse {
  static constexpr PacketHeader::Type kType = PacketHeader::kRegistrationResponse;

  player_id_t player_id;
};

struct GameThreadInitialization {
  static constexpr PacketHeader::Type kType = PacketHeader::kGameThreadInitialization;

  game_thread_id_t game_thread_id;
};

struct GameThreadInitializationResponse {
  static constexpr PacketHeader::Type kType = PacketHeader::kGameThreadInitializationResponse;

  int max_simultaneous_games;
};

struct StartGame {
  static constexpr PacketHeader::Type kType = PacketHeader::kStartGame;
  struct dynamic_size_section_t {
    char player_names[kSerializationLimit];  // names separated by null terminators
  };

  // Accepts an array of player names, and writes them to the dynamic_size_section.player_names buffer.
  // Also accepts the parent Packet as an argument, so that it can set the payload_size field.
  template<typename PacketT, size_t N> void load_player_names(
      PacketT& packet, const std::array<std::string, N>& player_names);

  // Inverse operation of load_player_names()
  template<size_t N> void parse_player_names(std::array<std::string, N>& player_names) const;

  game_id_t game_id;
  game_thread_id_t game_thread_id;
  player_id_t player_id;
  player_index_t seat_assignment;
  dynamic_size_section_t dynamic_size_section;
};

struct StateChange {
  static constexpr PacketHeader::Type kType = PacketHeader::kStateChange;
  struct dynamic_size_section_t {
    char buf[kSerializationLimit];
  };

  dynamic_size_section_t dynamic_size_section;
};

struct ActionPrompt {
  static constexpr PacketHeader::Type kType = PacketHeader::kActionPrompt;
  struct dynamic_size_section_t {
    char buf[kSerializationLimit];
  };

  dynamic_size_section_t dynamic_size_section;
};

struct Action {
  static constexpr PacketHeader::Type kType = PacketHeader::kAction;
  struct dynamic_size_section_t {
    char buf[kSerializationLimit];
  };

  dynamic_size_section_t dynamic_size_section;
};

using PayloadTypeList = mp::TypeList<
    Registration,
    RegistrationResponse,
    GameThreadInitialization,
    GameThreadInitializationResponse,
    StartGame,
    StateChange,
    ActionPrompt,
    Action>;
static_assert(mp::Length_v<PayloadTypeList> == PacketHeader::kNumTypes);

template <PacketPayloadConcept PacketPayload>
class Packet {
public:
  /*
   * Constructor initializes header but not the payload. The payload must be initialized separately.
   */
  Packet() : header_(PacketPayload::kType, sizeof(PacketPayload)) {}

  void set_dynamic_section_size(int buf_size);
  size_t size() const { return header_.payload_size + sizeof(PacketHeader); }
  const PacketHeader& header() const { return header_; }
  PacketHeader& header() { return header_; }
  const PacketPayload& payload() const { return payload_; }
  PacketPayload& payload() { return payload_; }

  void send_to(int socket_descriptor) const;
  void read_from(int socket_descriptor);

private:
  PacketHeader header_;
  PacketPayload payload_;
};

class GeneralPacket {
public:
  // The +1 helps to detect buffer-overflow on read()'s
  static constexpr int kMaxPayloadSize = mp::MaxSizeOf_v<PayloadTypeList> + 1;

  const PacketHeader& header() const { return header_; }

  /*
   * Checks that this packet is of the given type, and returns the payload reinterpreted to the given type. If the
   * check fails, throws an exception.
   */
  template<PacketPayloadConcept PacketPayload> const PacketPayload& to() const;

  void read_from(int socket_descriptor);

private:
  PacketHeader header_;
  char payload_[kMaxPayloadSize];
};

}  // namespace common

#include <common/inl/Packet.inl>
