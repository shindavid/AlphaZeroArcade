#pragma once

#include "core/BasicTypes.hpp"

namespace nim {

const int kNumPlayers = 2;
const int kMaxStonesToTake = 3;
const int kStartingStones = 21;

const core::action_t kTake1 = 0;
const core::action_t kTake2 = 1;
const core::action_t kTake3 = 2;
}  // namespace nim