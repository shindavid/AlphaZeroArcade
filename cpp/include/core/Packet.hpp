#pragma once

#include "core/BasicTypes.hpp"
#include "core/Constants.hpp"
#include "util/CppUtil.hpp"
#include "util/MetaProgramming.hpp"
#include "util/SocketUtil.hpp"

#include <array>
#include <concepts>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>

/*
 * Unit of data sent between GameServer and remote players.
 *
 * Changes to this file must be backwards-compatible, since we will test against frozen binaries
 * that were compiled using past versions of this file.
 */

namespace core {

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
    kActionDecision = 7,
    kEndGame = 8,
    kNumTypes = 9
  };

  Type type;
  int8_t padding[3] = {};
  int payload_size;

  PacketHeader(Type t = kNumTypes, int ps = 0) : type(t), payload_size(ps) {}
};
#pragma pack(pop)

static_assert(sizeof(PacketHeader) == 8);

namespace concepts {

template <class PayloadType>
concept PacketPayload = requires(PayloadType p) {
  { util::decay_copy(PayloadType::kType) } -> std::same_as<PacketHeader::Type>;
};

}  // namespace concepts

struct Registration {
  static constexpr PacketHeader::Type kType = PacketHeader::kRegistration;
  struct DynamicSizeSection {
    char player_name[kMaxNameLength + 1];  // +1 for null terminator
  };

  /*
   * If the remote process is registering multiple players, the server needs to allocate them all to
   * the same socket. Having this remaining_requests field allows the server to do this allocation
   * properly.
   */
  int remaining_requests;
  int max_simultaneous_games;
  seat_index_t requested_seat;  // negative = random seat
  DynamicSizeSection dynamic_size_section;
};

struct RegistrationResponse {
  static constexpr PacketHeader::Type kType = PacketHeader::kRegistrationResponse;
  struct DynamicSizeSection {
    char player_name[kMaxNameLength + 1];  // +1 for null terminator
  };

  player_id_t player_id;
  DynamicSizeSection dynamic_size_section;
};

struct GameThreadInitialization {
  static constexpr PacketHeader::Type kType = PacketHeader::kGameThreadInitialization;

  int num_game_slots;
};

struct GameThreadInitializationResponse {
  static constexpr PacketHeader::Type kType = PacketHeader::kGameThreadInitializationResponse;
};

struct StartGame {
  static constexpr PacketHeader::Type kType = PacketHeader::kStartGame;
  struct DynamicSizeSection {
    char player_names[kSerializationLimit];  // names separated by null terminators
  };

  // Accepts an array of player names, and writes them to the dynamic_size_section.player_names
  // buffer. Also accepts the parent Packet as an argument, so that it can set the payload_size
  // field.
  template <typename PacketT, size_t N>
  void load_player_names(PacketT& packet, const std::array<std::string, N>& player_names);

  // Inverse operation of load_player_names()
  template <size_t N>
  void parse_player_names(std::array<std::string, N>& player_names) const;

  game_id_t game_id;
  game_slot_index_t game_slot_index;
  player_id_t player_id;
  seat_index_t seat_assignment;
  DynamicSizeSection dynamic_size_section;
};

struct StateChange {
  static constexpr PacketHeader::Type kType = PacketHeader::kStateChange;
  struct DynamicSizeSection {
    char buf[kSerializationLimit];
  };

  game_slot_index_t game_slot_index;
  player_id_t player_id;
  DynamicSizeSection dynamic_size_section;
};

struct ActionPrompt {
  static constexpr PacketHeader::Type kType = PacketHeader::kActionPrompt;
  struct DynamicSizeSection {
    char buf[kSerializationLimit];
  };

  game_slot_index_t game_slot_index;
  player_id_t player_id;
  bool play_noisily;
  DynamicSizeSection dynamic_size_section;
};

struct ActionDecision {
  static constexpr PacketHeader::Type kType = PacketHeader::kActionDecision;
  struct DynamicSizeSection {
    char buf[kSerializationLimit];
  };

  game_slot_index_t game_slot_index;
  player_id_t player_id;
  DynamicSizeSection dynamic_size_section;
};

struct EndGame {
  static constexpr PacketHeader::Type kType = PacketHeader::kEndGame;
  struct DynamicSizeSection {
    char buf[kSerializationLimit];
  };

  game_slot_index_t game_slot_index;
  player_id_t player_id;
  DynamicSizeSection dynamic_size_section;
};

using PayloadTypeList = mp::TypeList<Registration, RegistrationResponse, GameThreadInitialization,
                                     GameThreadInitializationResponse, StartGame, StateChange,
                                     ActionPrompt, ActionDecision, EndGame>;
static_assert(mp::Length_v<PayloadTypeList> == PacketHeader::kNumTypes);

template <concepts::PacketPayload PacketPayload>
class Packet {
 public:
  /*
   * Constructor initializes header but not the payload. The payload must be initialized separately.
   */
  Packet() : header_(PacketPayload::kType, sizeof(PacketPayload)) {}

  /*
   * Assumes that PakcetPayload has a member dynamic_size_section, which has a single char buf[]
   * member player_name.
   *
   * Does a size-check on the name, and then copies it to that char buf[], calling
   * this->set_dynamic_section_size() properly.
   */
  void set_player_name(const std::string& name);

  void set_dynamic_section_size(int buf_size);
  size_t size() const { return header_.payload_size + sizeof(PacketHeader); }
  const PacketHeader& header() const { return header_; }
  PacketHeader& header() { return header_; }
  const PacketPayload& payload() const { return payload_; }
  PacketPayload& payload() { return payload_; }

  void send_to(io::Socket*) const;

  /*
   * Effectively a reinterpret_cast of the bytes on the socket to this.
   *
   * Returns false if the socket was shutdown.
   */
  bool read_from(io::Socket*);

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
   * Checks that this packet is of the given type, and returns the payload reinterpreted to the
   * given type. If the check fails, throws an exception.
   */
  template <concepts::PacketPayload PacketPayload>
  const PacketPayload& payload_as() const;

  /*
   * Effectively a reinterpret_cast of the bytes on the socket to this.
   *
   * Returns false if the socket was shutdown.
   */
  bool read_from(io::Socket*);

 private:
  PacketHeader header_;
  char payload_[kMaxPayloadSize];
};

}  // namespace core

#include "inline/core/Packet.inl"
