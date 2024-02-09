#include <games/carcassonne/BasicTypes.hpp>

#include <games/carcassonne/LookupTables.hpp>

namespace carcassonne {

inline std::string EdgeProfile::to_string() const {
  char buf[5];
  buf[0] = edge_type_to_string(get(Direction::dN))[0];
  buf[1] = edge_type_to_string(get(Direction::dE))[0];
  buf[2] = edge_type_to_string(get(Direction::dS))[0];
  buf[3] = edge_type_to_string(get(Direction::dW))[0];
  buf[4] = '\0';
  return buf;
}

inline EdgeType EdgeProfile::get(Direction direction) const {
  return EdgeType((data_ >> (6 - 2 * int(direction))) & 0x3);
}

inline std::string MeepleLocationProfile::to_string() const {
  char buf[64] = "";
  char* c = buf;
  for (int i = 0; i < 9; ++i) {
    if (permits(MeeplePlacement(i))) {
      c += sprintf(c, "%s|", meeple_placement_to_string(MeeplePlacement(i)));
    }
  }
  *(c-1) = '\0';  // remove trailing '|' character
  return buf;
}

inline bool MeepleLocationProfile::permits(MeeplePlacement m) const {
  return data_ & (1 << int(m));
}

inline const char* terrain_to_string(TerrainType terrain) {
  switch (terrain) {
    case TerrainType::tCloister: return "X";
    case TerrainType::tField: return "F";
    case TerrainType::tRoad: return "R";
    case TerrainType::tCity: return "C";
    default: return "?";
  }
}

inline const char* edge_type_to_string(EdgeType edge) {
  switch (edge) {
    case EdgeType::eNone: return "_";
    case EdgeType::eField: return "F";
    case EdgeType::eRoad: return "R";
    case EdgeType::eCity: return "C";
    default: return "?";
  }
}

inline const char* tile_type_to_string(TileType tile) {
  switch (tile) {
    case TileType::tCCCC110: return "CCCC110";
    case TileType::tCCCF100: return "CCCF100";
    case TileType::tCCCF110: return "CCCF110";
    case TileType::tCCCR100: return "CCCR100";
    case TileType::tCCCR110: return "CCCR110";
    case TileType::tCCFF100: return "CCFF100";
    case TileType::tCCFF110: return "CCFF110";
    case TileType::tCCFF200: return "CCFF200";
    case TileType::tCCRR100: return "CCRR100";
    case TileType::tCCRR110: return "CCRR110";
    case TileType::tCFCF100: return "CFCF100";
    case TileType::tCFCF110: return "CFCF110";
    case TileType::tCFCF200: return "CFCF200";
    case TileType::tCFFF100: return "CFFF100";
    case TileType::tCFRR100: return "CFRR100";
    case TileType::tCRFR100: return "CRFR100";
    case TileType::tCRRF100: return "CRRF100";
    case TileType::tCRRR100: return "CRRR100";
    case TileType::tFFFF001: return "FFFF001";
    case TileType::tFFFR001: return "FFFR001";
    case TileType::tFFRR000: return "FFRR000";
    case TileType::tFRFR000: return "FRFR000";
    case TileType::tFRRR000: return "FRRR000";
    case TileType::tRRRR000: return "RRRR000";
    default: return "???????";
  }
}

inline const char* direction_to_string(Direction direction) {
  switch (direction) {
    case Direction::dN: return "N";
    case Direction::dE: return "E";
    case Direction::dS: return "S";
    case Direction::dW: return "W";
    default: return "?";
  }
}

inline const char* meeple_placement_to_string(MeeplePlacement placement) {
  switch (placement) {
    case MeeplePlacement::mN: return "N";
    case MeeplePlacement::mE: return "E";
    case MeeplePlacement::mS: return "S";
    case MeeplePlacement::mW: return "W";
    case MeeplePlacement::mNE: return "NE";
    case MeeplePlacement::mSE: return "SE";
    case MeeplePlacement::mSW: return "SW";
    case MeeplePlacement::mNW: return "NW";
    case MeeplePlacement::mCloister: return "X";
    case MeeplePlacement::mNone: return "_";
    default: return "?";
  }
}

inline constexpr bool has_cloister(TileType tile) {
  return tile == TileType::tFFFF001 || tile == TileType::tFFFR001;
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
