#pragma once

#include <games/carcassonne/Constants.hpp>

#include <cstdint>

namespace carcassonne {

extern const uint32_t kValidOrientationTable[4];
extern const uint8_t kEdgeProfileTable[4 * kNumTileTypes];
extern const uint8_t kMeepleLocationProfileTable[4 * kNumTileTypes];

}  // namespace carcassonne

