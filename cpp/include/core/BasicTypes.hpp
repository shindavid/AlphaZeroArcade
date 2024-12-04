#pragma once

#include <util/CppUtil.hpp>
#include <util/FiniteGroups.hpp>

#include <cstdint>
#include <functional>
#include <tuple>

namespace core {

using seat_index_t = int8_t;
using player_id_t = int8_t;
using action_t = int32_t;
using game_id_t = int64_t;
using game_thread_id_t = int16_t;

}  // namespace core
