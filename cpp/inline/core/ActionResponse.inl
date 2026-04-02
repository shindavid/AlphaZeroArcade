#include "core/ActionResponse.hpp"

#include "util/Asserts.hpp"

namespace core {

template <concepts::Game Game>
template <typename T>
void ActionResponse<Game>::set_aux(T aux) {
  if constexpr (std::is_pointer_v<T>) {
    aux_ = reinterpret_cast<game_tree_node_aux_t>(aux);
  } else {
    // We are being explicit about the supported aux representations. Other types (e.g. char,
    // structs) could be supported in the future, but are intentionally disallowed for now.
    RELEASE_ASSERT(std::is_integral_v<T>,
                   "only integral and pointer types are supported for aux for now");
    aux_ = static_cast<game_tree_node_aux_t>(aux);
  }

  aux_set_ = true;
}

template <concepts::Game Game>
ActionResponse<Game>::ActionResponse(const Move& move) {
  set_move(move);
}

template <concepts::Game Game>
ActionResponse<Game> ActionResponse<Game>::construct(response_type_t type) {
  ActionResponse r;
  r.type_ = type;
  return r;
}

template <concepts::Game Game>
ActionResponse<Game> ActionResponse<Game>::backtrack(game_tree_index_t ix) {
  ActionResponse r = construct(kBacktrack);
  r.backtrack_node_ix_ = ix;
  return r;
}

template <concepts::Game Game>
ActionResponse<Game> ActionResponse<Game>::yield(int extra_enqueue_count) {
  ActionResponse r = construct(kYieldResponse);
  r.extra_enqueue_count_ = extra_enqueue_count;
  return r;
}

template <concepts::Game Game>
void ActionResponse<Game>::set_move(const Move& move) {
  type_ = kMakeMove;
  move_ = move;
}

template <concepts::Game Game>
core::yield_instruction_t ActionResponse<Game>::get_yield_instruction() const {
  switch (type_) {
    case kYieldResponse:
      return core::kYield;
    case kDropResponse:
      return core::kDrop;
    default:
      return core::kContinue;
  }
}

}  // namespace core
