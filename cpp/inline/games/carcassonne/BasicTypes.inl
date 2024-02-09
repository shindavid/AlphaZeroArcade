#include <games/carcassonne/BasicTypes.hpp>

#include <games/carcassonne/LookupTables.hpp>

namespace carcassonne {

inline constexpr bool has_cloister(TileType tile) {
  return tile == TileType::tFFFF001 || tile == TileType::tFFFR001;
}

inline EdgeType EdgeProfile::get(Direction direction) const {
  return EdgeType((data_ >> (6 - 2 * int(direction))) & 0x3);
}

inline bool MeepleLocationProfile::permits(MeeplePlacement m) const {
  return data_ & (1 << int(m));
}

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
