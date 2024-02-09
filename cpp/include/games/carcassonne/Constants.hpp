#pragma once

#include <core/BasicTypes.hpp>

namespace carcassonne {

constexpr int kNumPlayers = 2;
constexpr int kNumTiles = 72;
constexpr int kNumTileTypes = 24;
constexpr int kBoardDimension = 2 * kNumTiles - 1;

constexpr int kStartingX = kNumTiles - 1;
constexpr int kStartingY = kNumTiles - 1;

const core::seat_index_t kRed = 0;
const core::seat_index_t kBlue = 1;

}  // namespace carcassonne
