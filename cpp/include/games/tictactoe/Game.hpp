#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "core/GameRulesBase.hpp"
#include "core/GameTypes.hpp"
#include "core/IOBase.hpp"
#include "core/WinLossDrawResults.hpp"
#include "core/concepts/GameConcept.hpp"
#include "games/tictactoe/Constants.hpp"
#include "util/FiniteGroups.hpp"

#include <boost/functional/hash.hpp>

#include <functional>
#include <string>

namespace tictactoe {

constexpr mask_t make_mask(int a, int b, int c) {
  return (mask_t(1) << a) + (mask_t(1) << b) + (mask_t(1) << c);
}

/*
 * Bit order encoding for the board:
 *
 * 0 1 2
 * 3 4 5
 * 6 7 8
 */
class Game {
 public:
  struct Constants : public core::ConstantsBase {
    static constexpr const char* kGameName = "tictactoe";
    using kNumActionsPerMode = util::int_sequence<tictactoe::kNumCells>;
    static constexpr int kNumPlayers = tictactoe::kNumPlayers;
    static constexpr int kMaxBranchingFactor = tictactoe::kNumCells;
  };

  struct State {
    auto operator<=>(const State& other) const = default;
    size_t hash() const;
    mask_t opponent_mask() const { return full_mask ^ cur_player_mask; }
    core::seat_index_t get_player_at(int row, int col) const;

    mask_t full_mask;        // spaces occupied by either player
    mask_t cur_player_mask;  // spaces occupied by current player
  };

  using GameResults = core::WinLossDrawResults;
  using SymmetryGroup = groups::D4;
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;

  struct Symmetries {
    static Types::SymmetryMask get_mask(const State& state);
    static void apply(State& state, group::element_t sym);
    static void apply(Types::PolicyTensor& policy, group::element_t sym, core::action_mode_t = 0);
    static void apply(core::action_t& action, group::element_t sym, core::action_mode_t = 0);
    static group::element_t get_canonical_symmetry(const State& state);
  };

  struct Rules : public core::RulesBase<Types> {
    static void init_state(State&);
    static Types::ActionMask get_legal_moves(const State&);
    static core::action_mode_t get_action_mode(const State&) { return 0; }
    static core::seat_index_t get_current_player(const State&);
    static void apply(State&, core::action_t action);
    static bool is_terminal(const State& state, core::seat_index_t last_player,
                            core::action_t last_action, GameResults::Tensor& outcome);
  };

  struct IO : public core::IOBase<Types> {
    static std::string action_delimiter() { return ""; }
    static std::string action_to_str(core::action_t action, core::action_mode_t) {
      return std::to_string(action);
    }
    static std::string player_to_str(core::seat_index_t player) {
      return (player == tictactoe::kX) ? "X" : "O";
    }
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);
    static std::string compact_state_repr(const State& state);

    static boost::json::value state_to_json(const State& state);
  };

  static constexpr mask_t kThreeInARowMasks[] = {
    make_mask(0, 1, 2), make_mask(3, 4, 5), make_mask(6, 7, 8), make_mask(0, 3, 6),
    make_mask(1, 4, 7), make_mask(2, 5, 8), make_mask(0, 4, 8), make_mask(2, 4, 6)};

  static void static_init() {}
};

}  // namespace tictactoe

namespace std {

template <>
struct hash<tictactoe::Game::State> {
  size_t operator()(const tictactoe::Game::State& pos) const { return pos.hash(); }
};

}  // namespace std

static_assert(core::concepts::Game<tictactoe::Game>);

#include "inline/games/tictactoe/Game.inl"

// Ensure that we always have bindings when we #include "games/tictactoe/Game.hpp":
#include "games/tictactoe/Bindings.hpp"  // IWYU pragma: keep
