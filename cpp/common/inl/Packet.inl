#include <common/Packet.hpp>

#include <sys/socket.h>

#include <boost/lexical_cast.hpp>

#include <util/Exception.hpp>

namespace common {

template<typename PacketT, size_t N>
void StartGame::load_player_names(PacketT& packet, const std::array<std::string, N>& player_names) {
  constexpr int max_total_name_length = N * (kMaxNameLength + 1);  // + 1 for null terminator
  constexpr int buf_size = sizeof(dynamic_size_section.player_names);
  static_assert(max_total_name_length <= buf_size, "StartGame player_names buffer too small");

  char* p = dynamic_size_section.player_names;
  for (const std::string& name : player_names) {
    util::clean_assert(name.size() <= kMaxNameLength, "StartGame::load_player_names() name too long [\"%s\"] (%d > %d)",
                       name.c_str(), (int)name.size(), kMaxNameLength);
    memcpy(p, name.c_str(), name.size() + 1);
    p += name.size() + 1;
  }
  int bytes_written = p - dynamic_size_section.player_names;
  packet.set_dynamic_section_size(bytes_written);
}

template<size_t N>
void StartGame::parse_player_names(std::array<std::string, N>& player_names) const {
  constexpr int max_total_name_length = N * (kMaxNameLength + 1);  // + 1 for null terminator
  constexpr int buf_size = sizeof(dynamic_size_section.player_names);
  static_assert(max_total_name_length <= buf_size, "StartGame player_names buffer too small");

  const char* p = dynamic_size_section.player_names;
  for (size_t i = 0; i < N; ++i) {
    player_names[i] = p;
    int n = player_names[i].size();

    util::clean_assert(n > 0, "StartGame::parse_player_names() empty name (i=%ld)", i);
    util::clean_assert(n <= kMaxNameLength, "StartGame::parse_player_names() name too long [\"%s\"] (i=%ld) (%d > %d)",
                       p, i, n, kMaxNameLength);
    p += player_names[i].size() + 1;
  }
}

template <PacketPayloadConcept PacketPayload>
void Packet<PacketPayload>::set_dynamic_section_size(int buf_size) {
  constexpr int orig_size = sizeof(typename PacketPayload::dynamic_size_section_t);
  if (buf_size < 0 || buf_size > orig_size) {
    throw util::Exception("Packet<%d>::set_dynamic_section_size() invalid buf_size (%d) orig_size=%d",
                          (int)PacketPayload::kType, buf_size, orig_size);
  }
  header_.payload_size = sizeof(PacketPayload) - orig_size + buf_size;
}

template <PacketPayloadConcept PacketPayload>
void Packet<PacketPayload>::send_to(int socket_descriptor) const {
  send(socket_descriptor, (const void*) this, size(), 0);
}

template <PacketPayloadConcept PacketPayload>
void Packet<PacketPayload>::read_from(int socket_descriptor) {
  char* buf = reinterpret_cast<char*>(this);
  constexpr int buf_size = sizeof(*this);
  int n = recv(socket_descriptor, buf, buf_size, 0);
  if (n <= 0) {
    throw util::Exception("Packet<%d>::read_from() failed (n=%d)", (int)PacketPayload::kType, n);
  }
  if (PacketPayload::kType != header_.type) {
    throw util::Exception("Packet<%d>::read_from() invalid type (expected:%d, got:%d)",
                          (int)PacketPayload::kType, (int)PacketPayload::kType, (int)header_.type);
  }
}

template<PacketPayloadConcept PacketPayload> const PacketPayload& GeneralPacket::to() const {
  if (header_.type != PacketPayload::kType) {
    throw util::Exception("Packet::to() invalid type (expected:%d, got:%d)",
                          (int)PacketPayload::kType, (int)header_.type);
  }
  return *reinterpret_cast<const PacketPayload*>(payload_);
}

inline void GeneralPacket::read_from(int socket_descriptor) {
  char* buf = reinterpret_cast<char*>(this);
  constexpr int buf_size = sizeof(*this);
  int n = recv(socket_descriptor, buf, buf_size, 0);
  if (n <= 0) {
    throw util::Exception("GeneralPacket::read_from() failed (n=%d)", n);
  }
  if (n >= buf_size) {
    throw util::Exception("GeneralPacket::read_from() potential buffer overflow (%d >= %d)", n, buf_size);
  }
}

}  // namespace common
