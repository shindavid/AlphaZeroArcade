#pragma once

#include <util/CppUtil.hpp>
#include <util/FiniteGroups.hpp>

#include <cstdint>

namespace core {

using seat_index_t = int8_t;
using player_id_t = int8_t;
using action_mode_t = int8_t;
using action_t = int32_t;
using game_id_t = int64_t;
using game_thread_id_t = int16_t;
using game_slot_index_t = int16_t;
using search_context_id_t = int16_t;
using nn_evaluation_pool_index_t = int32_t;
using nn_evaluation_sequence_id_t = int64_t;

// yield_instruction_t is used in various components of the MCTS machinery
//
// kYield indicates that the current thread should yield its execution to the next thread.
//
// kContinue indicates that the current thread should continue its execution without yielding.
enum yield_instruction_t : int8_t {
  kContinue,
  kYield
};

}  // namespace core
