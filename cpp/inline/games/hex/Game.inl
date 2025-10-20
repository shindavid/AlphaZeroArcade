#include "games/hex/Game.hpp"

#include "core/DefaultCanonicalizer.hpp"
#include "games/hex/Constants.hpp"
#include "util/AnsiCodes.hpp"
#include "util/EigenUtil.hpp"

#include <bit>

namespace hex {

inline void Game::Symmetries::apply(State& state, group::element_t sym) {
  switch (sym) {
    case groups::C2::kIdentity:
      return;
    case groups::C2::kRot180:
      return state.rotate();
    default:
      throw util::Exception("Unknown group element: {}", sym);
  }
}

template<eigen_util::concepts::FTensor Tensor>
void Game::Symmetries::apply(Tensor& tensor, group::element_t sym, core::action_mode_t) {
  constexpr int N = Constants::kBoardDim;
  switch (sym) {
    case groups::C2::kIdentity:
      return;
    case groups::C2::kRot180:
      return eigen_util::rot180<N>(tensor);
    default:
      throw util::Exception("Unknown group element: {}", sym);
  }
}

inline void Game::Symmetries::apply(core::action_t& action, group::element_t sym,
                                    core::action_mode_t) {
  switch (sym) {
    case groups::C2::kIdentity:
      return;
    case groups::C2::kRot180: {
      action = action == kSwap ? kSwap : (Constants::kNumSquares - 1 - action);
      return;
    }
    default:
      throw util::Exception("Unknown group element: {}", sym);
  }
}

inline group::element_t Game::Symmetries::get_canonical_symmetry(const State& state) {
  using DefaultCanonicalizer = core::DefaultCanonicalizer<Game>;
  return DefaultCanonicalizer::get(state);
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const State& state) {
  const State::Core& core = state.core;
  Types::ActionMask valid_actions;

  int offset = 0;
  for (int i = 0; i < Constants::kBoardDim; ++i) {
    mask_t occupied_mask = core.rows[Constants::kRed][i] | core.rows[Constants::kBlue][i];
    mask_t free_mask = ~occupied_mask & ((mask_t(1) << Constants::kBoardDim) - 1);
    for (; free_mask; free_mask &= free_mask - 1) {
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

inline core::action_t Game::Rules::compute_mirror_action(core::action_t action) {
  static constexpr auto B = Constants::kBoardDim;
  int8_t row = action / B;
  int8_t col = action % B;
  return B * col + row;
}

inline std::string Game::IO::action_to_str(core::action_t action, core::action_mode_t) {
  if (action == kSwap) {
    return "swap";
  }
  int row = action / Constants::kBoardDim;
  int col = action % Constants::kBoardDim;
  return std::format("{:c}{}", 'A' + col, row + 1);
}

inline std::string Game::IO::player_to_str(core::seat_index_t player) {
  return (player == Constants::kBlue)
           ? std::format("{}{}{}", ansi::kBlue(""), ansi::kCircle("B"), ansi::kReset(""))
           : std::format("{}{}{}", ansi::kRed(""), ansi::kCircle("R"), ansi::kReset(""));
}

}  // namespace hex
