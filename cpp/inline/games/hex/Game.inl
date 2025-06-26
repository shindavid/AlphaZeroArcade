#include <games/hex/Game.hpp>

#include <util/AnsiCodes.hpp>
#include <util/Asserts.hpp>
#include <util/CppUtil.hpp>

#include <bit>

namespace hex {

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const StateHistory& history) {
  return get_legal_moves(history.current());
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const State& state) {
  const State::Core& core = state.core;
  Types::ActionMask valid_actions;

  int offset = 0;
  for (int i = 0; i < Constants::kBoardDim; ++i) {
    mask_t occupied_mask = core.rows[Constants::kBlack][i] | core.rows[Constants::kWhite][i];
    mask_t free_mask = ~occupied_mask & ((mask_t(1)<<Constants::kBoardDim)-1);
    for (; free_mask; free_mask &= free_mask-1) {
      int j = std::countr_zero(free_mask);
      valid_actions[offset + j] = true;
    }

    offset += Constants::kBoardDim;
  }

  if (core.cur_player == Constants::kSecondPlayer && !core.post_swap_phase) {
    valid_actions[kSwap] = true;
  }
  return valid_actions;
}

inline bool Game::Rules::is_terminal(const State& state, core::seat_index_t last_player,
                                     core::action_t last_action, GameResults::Tensor& outcome) {
  const auto& U = state.aux.union_find[last_player];
  if (U.connected(UnionFind::kVirtualVertex1, UnionFind::kVirtualVertex2)) {
    outcome = GameResults::win(last_player);
    return true;
  }
  return false;
}

inline std::string Game::IO::action_to_str(core::action_t action, core::action_mode_t) {
  if (action == kSwap) {
    return "swap";
  }
  int row = action / Constants::kBoardDim;
  int col = action % Constants::kBoardDim;
  return std::format("{}{}", 'A' + row, col + 1);
}

inline std::string Game::IO::player_to_str(core::seat_index_t player) {
  return (player == Constants::kWhite)
             ? std::format("{}{}{}", ansi::kWhite(""), ansi::kCircle("W"), ansi::kReset(""))
             : std::format("{}{}{}", ansi::kBlue(""), ansi::kCircle("B"), ansi::kReset(""));
}

template <typename Iter>
Game::InputTensorizor::Tensor Game::InputTensorizor::tensorize(Iter start, Iter cur) {
  constexpr int B = Constants::kBoardDim;
  constexpr int P = Constants::kNumPlayers;
  core::seat_index_t cp = Rules::get_current_player(*cur);
  Tensor tensor;
  tensor.setZero();
  int i = 0;
  Iter it = cur;
  while (true) {
    State& state = *it;
    for (int p = 0; p < P; ++p) {
      for (int row = 0; row < B; ++row) {
        mask_t mask = state.core.rows[p][row];
        for (; mask; mask &= mask - 1) {
          int col = std::countr_zero(mask);
          int x = (P + cp - p) % P;
          tensor(i + x, row, col) = 1;
        }
      }
    }
    if (it == start) break;
    it--;
    i += P;
  }

  if (cp == Constants::kWhite && !cur->core.post_swap_phase) {
    // add swap legality plane
    tensor.chip(i, 0).setConstant(1);
  }

  return tensor;
}

}  // namespace hex
