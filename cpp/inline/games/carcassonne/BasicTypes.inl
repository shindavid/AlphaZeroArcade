#include <games/carcassonne/BasicTypes.hpp>

#include <games/carcassonne/LookupTables.hpp>

namespace carcassonne {

inline constexpr bool has_cloister(TileType tile) {
  return tile == TileType::tFFFF001 || tile == TileType::tFFFR001;
}

inline EdgeType EdgeProfile::get(Direction direction) const {
  return EdgeType((data_ >> (6 - 2 * int(direction))) & 0x3);
}

// inline void EdgeType::rotate(Direction orientation) {
//   int k = 2 * int(orientation);
//   uint8_t x = data_ >> k;
//   uint8_t y = data_ << (8 - k);
//   data_ = x + y;
// }

inline bool MeepleLocationProfile::permits(MeeplePlacement m) const {
  return data_ & (1 << int(m));
}

// inline void MeepleLocationProfile::rotate(Direction orientation) {
//   uint8_t high = (data_ >> 4) & 0xF;
//   uint8_t low = data_ & 0xF;

//   high = rotate_4bits_left(high, int(orientation));
//   low = rotate_4bits_left(low, int(orientation));

//   data_ &= ~0xFF;
//   data_ += (high << 4) + low;
// }

// static inline uint8_t MeepleLocationProfile::rotate_4bits_left(uint8_t x, int k) {
//   uint8_t y = x << k;
//   return (y & 0xF) + (y >> 4);
// }

inline bool is_valid_orientation(TileType tile, Direction orientation) {
  return kValidOrientationTable[int(orientation)] & (1 << int(tile));
}

inline EdgeProfile get_edge_profile(TileType tile, Direction orientation) {
  EdgeProfile profile(kEdgeProfileTable[4 * int(tile) + int(orientation)]);
  return profile;
}

inline MeepleLocationProfile get_valid_meeple_placements(TileType tile, Direction orientation) {
  uint16_t data = kMeepleLocationProfileTable[4 * int(tile) + int(orientation)];
  bool c = has_cloister(tile);

  constexpr uint16_t none = 1 << int(MeeplePlacement::mNone);
  constexpr uint16_t cloister = 1 << int(MeeplePlacement::mCloister);
  constexpr uint16_t extra[2] = {none, none | cloister};

  MeepleLocationProfile profile(data + extra[c]);
  return profile;
}

}  // namespace carcassonne
