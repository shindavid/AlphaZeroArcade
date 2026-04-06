#pragma once

#include "core/BasicTypes.hpp"
#include "games/chess/TypeDefs.hpp"

#include <cstdint>

namespace a0achess {

const int kNumPlayers = 2;
const int kBoardDim = 8;

const core::seat_index_t kWhite = 0;
const core::seat_index_t kBlack = 1;

const int kNumMoves = 1858;           // From lc0
const int kMaxBranchingFactor = 256;  // What Disservin uses

const int kNumPastFramesToEncode = 7;
constexpr int kNumRecentHashesToStore = 8;

const board_mask_t kPawnsMask = 0x00FFFFFFFFFFFF00;

constexpr uint64_t kHistoryHashRollConstant = 0x9e3779b97f4a7c15UL;

}  // namespace a0achess
