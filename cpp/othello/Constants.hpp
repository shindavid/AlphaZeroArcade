#pragma once

#include <cstdint>

#include <common/BasicTypes.hpp>

namespace othello {

using column_t = int8_t;
using row_t = int8_t;
using mask_t = uint64_t;

const int kNumPlayers = 2;
const int kBoardDimension = 8;
const int kBoardSize = kBoardDimension * kBoardDimension;
const int kNumStartingPieces = 4;

/*
 * Technically, we can leave out the 4 starting positions, but that makes the mapping of board position to action index
 * inelegant.
 */
const int kNumGlobalActions = kBoardSize;

/*
 * This can probably be shrunk, maybe down to to the 22-34 range. But I haven't found any proof of an upper bound, so
 * being conservative for now.
 *
 * https://puzzling.stackexchange.com/a/102017/18525
 */
const int kMaxNumLocalActions = kNumGlobalActions - kNumStartingPieces;

const int kTypicalNumMovesPerGame = kBoardSize - kNumStartingPieces;

const common::seat_index_t kBlack = 0;
const common::seat_index_t kWhite = 1;

}  // namespace othello
