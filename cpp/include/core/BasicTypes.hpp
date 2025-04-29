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
using context_id_t = int16_t;
using nn_evaluation_pool_index_t = int32_t;
using nn_evaluation_sequence_id_t = int64_t;

// yield_instruction_t is used in various components of the MCTS machinery
//
// kContinue indicates that the current thread can continue executing.
//
// kYield indicates that the current thread is blocked waiting for some asynchronous event to
// finish (such as the completion of an NN evaluation). It is requesting the caller to move onto
// a different unit of work. When the caller comes back to this thread, it is required to return
// kContinue. Thus, kYield should only be used when the block is expected to be short-lived.
//
// kHibernate is similar to kYield, but indicates that the block is expected to be long-lived. When
// GameServer receives a kHibernate response, it does not cycle back to this thread on its own, and
// instead awaits an explicit notification from the thread that its hibernation is over. An example
// use case for this is for a human player waiting for a TUI input. We don't know how long the
// human will take to respond, so we don't want to waste CPU cycles polling for it.
//
// kDrop indicates that the current thread should be dropped. This is used in the context of
// multithreaded search: the first search thread to process a state will return kYield with
// extra_enqueue_count = n > 0. This spawns n extra threads to process the same state. When the
// multi-threaded search is complete, the thread that finishes the job will return kContinue, and
// the other n will return kDrop. The GameServer will then drop the n threads, going back to only
// have one copy of the GameSlot in the queue.
enum yield_instruction_t : int8_t {
  kContinue,
  kYield,
  kHibernate,
  kDrop
};

}  // namespace core
