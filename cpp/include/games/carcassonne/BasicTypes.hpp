#pragma once

#include <cstdint>

namespace carcassonne {

enum class TerrainType : uint8_t {
  tField,
  tCloister,
  tRoad,
  tCity
};

enum class EdgeType : uint8_t {
  eNone,
  eField,
  eRoad,
  eCity
};

/*
 * Naming:
 *
 * tXXXXabc
 *
 * The X's name the edge types in clockwise order (F, R, and C stand for field, road, and city,
 * respectively).
 *
 * The a, b, and c are the number of cities, shields, and cloisters, respectively.
 *
 * There can be multiple possible ways to name the XXXX part, due to rotations. We always choose
 * the alphabetically smallest representation.
 */
enum class TileType : uint8_t {
  tCCCC110,
  tCCCF100,
  tCCCF110,
  tCCCR100,
  tCCCR110,
  tCCFF100,
  tCCFF110,
  tCCFF200,
  tCCRR100,
  tCCRR110,
  tCFCF100,
  tCFCF110,
  tCFCF200,
  tCFFF100,
  tCFRR100,
  tCRFR100,
  tCRRF100,
  tCRRR100,
  tFFFF001,
  tFFFR001,
  tFFRR000,
  tFRFR000,
  tFRRR000,
  tRRRR000
};

inline constexpr bool has_cloister(TileType tile);

/*
 * The four cardinal directions.
 *
 * When used in conjunction with a TileType, this specifies a tile orientation. The TileType names
 * the 4 edge types in clockwise order. The Direction specifies the orientation of the first edge.
 *
 * For example, if the edge is tCRRR10, and the orientation is dE, then the city (C) points east.
 */
enum class Direction : uint8_t {
  dN = 0,
  dE = 1,
  dS = 2,
  dW = 3
};

/*
 * A meeple can be placed in a field, road, city, or cloister.
 *
 * We capture the set of {field, road, city} choices with the 8 cardinal/diagonal directions.
 * The diagonal directions are used for fields, and the cardinal directions are used for roads and
 * cities.
 *
 * Note that for certain (most) tile types, two or more of the cardinal/diagonal directions will
 * map to the same terrain region. The function get_valid_meeple_placements() arbitrarily
 * decides which of the equivalent placements is valid, so that move representations are unique.
 */
enum class MeeplePlacement : uint8_t {
  mN = 0,
  mE = 1,
  mS = 2,
  mW = 3,
  mNE = 4,
  mSE = 5,
  mSW = 6,
  mNW = 7,
  mCloister = 8,
  mNone = 9
};

/*
 * An EdgeProfile is effectively a 4-element lookup table, mapping each Direction (dN, dE, dS, dW)
 * to an EdgeType (eField, eRoad, eCity).
 *
 * Each TileType/Direction pair produces a unique EdgeProfile.
 */
class __attribute__((packed)) EdgeProfile {
 public:
  EdgeProfile(uint8_t data) : data_(data) {}
  EdgeType get(Direction direction) const;

 private:
  uint8_t data_;  // high bits to low bits: NESW
};

/*
 * A set of valid meeple locations.
 *
 * Each TileType/Direction pair produces a unique MeepleLocationProfile.
 *
 * For a given tile, one or more of the candidate MeeplePlacement's may be equivalent. In such
 * cases, one of the placements is chosen as the canonical representative, and the others are
 * considered invalid.
 */
class __attribute__((packed)) MeepleLocationProfile {
 public:
  MeepleLocationProfile(uint16_t data) : data_(data) {}

  bool permits(MeeplePlacement m) const;

 private:
  const uint16_t data_;  // bit k corresponds to MeeplePlacement k
};

/*
 * Is the given orientation valid for the given tile?
 *
 * Certain orientations are considered invalid because they are equivalent to other orientations.
 */
bool is_valid_orientation(TileType tile, Direction orientation);

/*
 * If tile is rotated to the given orientation, what is the resultant edge profile?
 */
EdgeProfile get_edge_profile(TileType tile, Direction orientation);

/*
 * If placing a tile in the given orientation, where are the valid meeple placements?
 */
MeepleLocationProfile get_valid_meeple_placements(TileType tile, Direction orientation);

}  // namespace carcassonne

#include <inline/games/carcassonne/BasicTypes.inl>
