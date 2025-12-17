#pragma once

#include "util/AllocPool.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <cstdint>
#include <memory>
#include <queue>
#include <vector>

namespace core {

using seat_index_t = int8_t;
using player_id_t = int8_t;
using action_mode_t = int8_t;
using action_t = int32_t;
using game_id_t = int64_t;
using game_thread_id_t = int16_t;
using nn_evaluation_pool_index_t = int32_t;
using nn_evaluation_sequence_id_t = int64_t;
using pipeline_index_t = int32_t;
using generation_t = int32_t;
using node_aux_t = uint64_t;
using node_ix_t = int32_t;

// A game_slot_index_t specifies a GameSlot within a GameServer/GameServerProxy.
using game_slot_index_t = int16_t;

// Each player can manage 1 or more "Contexts". This is a notion that is internal to each player.
// For single-threaded players, this is always 0. For multi-threaded players, this can be thought of
// as a zero-indexed thread-id.
//
// The GameServer paradigm allows a player to inform the GameServer that it will use multiple
// contexts (i.e., threads). The GameServer needs this information from the player in order to
// optimally schedule work across the set of all parallel games.
using context_id_t = int16_t;

// SlotContext is essentially a (game_slot_index_t, context_id_t) pair.
struct SlotContext {
  SlotContext(game_slot_index_t s = -1, context_id_t c = 0) : slot(s), context(c) {}

  game_slot_index_t slot;
  context_id_t context;
};
using slot_context_vec_t = std::vector<SlotContext>;
using slot_context_queue_t = std::queue<SlotContext>;

// yield_instruction_t is used in various components of the MCTS machinery
//
// kContinue indicates that the current thread can continue executing.
//
// kYield indicates that the current thread is blocked waiting for some asynchronous event to
// finish (such as the completion of an NN evaluation). It is requesting the caller to move onto
// a different unit of work. When the asynchronous event is completed, the GameServer will be
// notified, at which point it can put the GameSlot back in its queue.
//
// kDrop indicates that the current thread should be dropped. This is used in the context of
// multithreaded search: the first search thread to process a state will return kYield with
// extra_enqueue_count = n > 0. This spawns n extra threads to process the same state. When the
// multi-threaded search is complete, the thread that finishes the job will return kContinue, and
// the other n will return kDrop. The GameServer will then drop the n threads, going back to only
// have one copy of the GameSlot in the queue.
enum yield_instruction_t : int8_t { kContinue, kYield, kDrop };

using mutex_vec_t = std::vector<mit::mutex>;
using mutex_vec_sptr_t = std::shared_ptr<mutex_vec_t>;

using node_pool_index_t = util::pool_index_t;
using edge_pool_index_t = util::pool_index_t;

}  // namespace core
