#pragma once

#include <cstdint>

namespace blokus {

using piece_id_t = int8_t;

const int kNumPlayers = 4;
const int kBoardDimension = 20;
const int kNumCells = kBoardDimension * kBoardDimension;

}  // namespace blokus
