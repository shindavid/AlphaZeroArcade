#include <core/Packet.hpp>

#include <sys/socket.h>

#include <boost/lexical_cast.hpp>

#include <util/Asserts.hpp>
#include <util/Exception.hpp>

namespace core {

template <typename PacketT, size_t N>
void StartGame::load_player_names(PacketT& packet, const std::array<std::string, N>& player_names) {
  constexpr int max_total_name_length = N * (kMaxNameLength + 1);  // + 1 for null terminator
  constexpr int buf_size = sizeof(dynamic_size_section.player_names);
  static_assert(max_total_name_length <= buf_size, "StartGame player_names buffer too small");

  char* p = dynamic_size_section.player_names;
  for (const std::string& name : player_names) {
    util::clean_assert(name.size() <= kMaxNameLength,
                       "StartGame::load_player_names() name too long [\"%s\"] (%d > %d)",
                       name.c_str(), (int)name.size(), kMaxNameLength);
    memcpy(p, name.c_str(), name.size() + 1);
    p += name.size() + 1;
  }
  int bytes_written = p - dynamic_size_section.player_names;
  packet.set_dynamic_section_size(bytes_written);
}

template <size_t N>
void StartGame::parse_player_names(std::array<std::string, N>& player_names) const {
  constexpr int max_total_name_length = N * (kMaxNameLength + 1);  // + 1 for null terminator
  constexpr int buf_size = sizeof(dynamic_size_section.player_names);
  static_assert(max_total_name_length <= buf_size, "StartGame player_names buffer too small");

  const char* p = dynamic_size_section.player_names;
  for (size_t i = 0; i < N; ++i) {
    player_names[i] = p;
    int n = player_names[i].size();

    util::clean_assert(n > 0, "StartGame::parse_player_names() empty name (i=%ld)", i);
    util::clean_assert(n <= kMaxNameLength,
                       "StartGame::parse_player_names() name too long [\"%s\"] (i=%ld) (%d > %d)",
                       p, i, n, kMaxNameLength);
    p += player_names[i].size() + 1;
  }
}

template <concepts::PacketPayload PacketPayload>
void Packet<PacketPayload>::set_player_name(const std::string& name) {
  constexpr int buf_size = sizeof(payload_.dynamic_size_section.player_name);
  if (name.size() + 1 > buf_size) {  // + 1 for null terminator
    throw util::Exception("Packet<%d>::set_player_name() name too long [\"%s\"] (%d + 1 > %d)",
                          (int)PacketPayload::kType, name.c_str(), (int)name.size(), buf_size);
  }
  memcpy(payload_.dynamic_size_section.player_name, name.c_str(), name.size() + 1);
  set_dynamic_section_size(name.size() + 1);  // + 1 for null terminator
}

template <concepts::PacketPayload PacketPayload>
void Packet<PacketPayload>::set_dynamic_section_size(int buf_size) {
  constexpr int orig_size = sizeof(typename PacketPayload::dynamic_size_section_t);
  if (buf_size < 0 || buf_size > orig_size) {
    throw util::Exception(
        "Packet<%d>::set_dynamic_section_size() invalid buf_size (%d) orig_size=%d",
        (int)PacketPayload::kType, buf_size, orig_size);
  }
  header_.payload_size = sizeof(PacketPayload) - orig_size + buf_size;
}

template <concepts::PacketPayload PacketPayload>
void Packet<PacketPayload>::send_to(io::Socket* socket) const {
  socket->write((const char*)this, size());
}

template <concepts::PacketPayload PacketPayload>
bool Packet<PacketPayload>::read_from(io::Socket* socket) {
  char* buf = reinterpret_cast<char*>(this);
  constexpr int header_size = sizeof(header_);

  io::Socket::Reader reader(socket);
  if (!reader.read(buf, header_size)) {
    return false;
  }

  if (PacketPayload::kType != header_.type) {
    throw util::Exception("Packet<%d>::read_from() invalid type (expected:%d, got:%d)",
                          (int)PacketPayload::kType, (int)PacketPayload::kType, (int)header_.type);
  }
  if (header_.payload_size > (int)sizeof(payload_)) {
    throw util::Exception("Packet<%d>::read_from() invalid payload_size (%d>%d)",
                          (int)PacketPayload::kType, header_.payload_size, (int)sizeof(payload_));
  }

  if (!reader.read(buf + header_size, header_.payload_size)) {
    return false;
  }
  return true;
}

template <concepts::PacketPayload PacketPayload>
const PacketPayload& GeneralPacket::payload_as() const {
  if (header_.type != PacketPayload::kType) {
    throw util::Exception("GeneralPacket::payload_as() invalid type (expected:%d, got:%d)",
                          (int)PacketPayload::kType, (int)header_.type);
  }
  return *reinterpret_cast<const PacketPayload*>(payload_);
}

inline bool GeneralPacket::read_from(io::Socket* socket) {
  char* buf = reinterpret_cast<char*>(this);
  constexpr int header_size = sizeof(header_);

  io::Socket::Reader reader(socket);
  if (!reader.read(buf, header_size)) {
    return false;
  }

  if (header_.payload_size > (int)sizeof(payload_)) {
    throw util::Exception("GeneralPacket::read_from() invalid payload_size (%d>%d)",
                          header_.payload_size, (int)sizeof(payload_));
  }

  if (header_.payload_size > 0 && !reader.read(buf + header_size, header_.payload_size)) {
    return false;
  }
  return true;
}

}  // namespace core
