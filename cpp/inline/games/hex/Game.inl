#include "games/hex/Game.hpp"

#include "core/BasicTypes.hpp"
#include "games/hex/Constants.hpp"
#include "util/AnsiCodes.hpp"

#include <bit>

namespace hex {

inline Game::Rules::Result Game::Rules::analyze(const State& state) {
  core::seat_index_t last_player = 1 - state.core.cur_player;
  if (last_player >= 0) {
    const auto& U = state.aux.union_find[last_player];
    if (U.connected(UnionFind::kVirtualVertex1, UnionFind::kVirtualVertex2)) {
      return PlayerResult::make_win<Constants::kNumPlayers>(last_player);
    }
  }

  const State::Core& core = state.core;
  Types::MoveSet valid_moves;

  int offset = 0;
  for (int i = 0; i < Constants::kBoardDim; ++i) {
    mask_t occupied_mask = core.rows[Constants::kRed][i] | core.rows[Constants::kBlue][i];
    mask_t free_mask = ~occupied_mask & ((mask_t(1) << Constants::kBoardDim) - 1);
    for (; free_mask; free_mask &= free_mask - 1) {
      int j = std::countr_zero(free_mask);
      valid_moves.add(offset + j);
    }

    offset += Constants::kBoardDim;
  }

  if (core.cur_player == Constants::kSecondPlayer && !core.post_swap_phase) {
    valid_moves.add(kSwap);
  }

  return valid_moves;
}

inline std::string Game::IO::player_to_str(core::seat_index_t player) {
  return (player == Constants::kBlue)
           ? std::format("{}{}{}", ansi::kBlue(""), ansi::kCircle("B"), ansi::kReset(""))
           : std::format("{}{}{}", ansi::kRed(""), ansi::kCircle("R"), ansi::kReset(""));
}

}  // namespace hex
