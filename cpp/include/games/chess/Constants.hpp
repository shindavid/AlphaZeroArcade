#pragma once

#include "core/BasicTypes.hpp"

namespace chess {

const int kNumPlayers = 2;
const int kBoardDim = 8;

const core::seat_index_t kWhite = 0;
const core::seat_index_t kBlack = 1;

const int kNumActions = 1858;         // From lc0
const int kMaxBranchingFactor = 500;  // ChatGPT estimates 250, doubling to be generous

const int kNumPastStatesToEncode = 7;

}  // namespace chess
