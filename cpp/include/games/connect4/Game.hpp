#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "core/GameTypes.hpp"
#include "core/IOBase.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/SimpleStateHistory.hpp"
#include "core/WinLossDrawResults.hpp"
#include "core/concepts/Game.hpp"
#include "games/GameRulesBase.hpp"
#include "games/connect4/Constants.hpp"
#include "util/CppUtil.hpp"
#include "util/FiniteGroups.hpp"

#include <boost/functional/hash.hpp>

#include <functional>
#include <string>

namespace c4 {

/*
 * Bit order encoding for the board:
 *
 *  5 13 21 29 37 45 53
 *  4 12 20 28 36 44 52
 *  3 11 19 27 35 43 51
 *  2 10 18 26 34 42 50
 *  1  9 17 25 33 41 49
 *  0  8 16 24 32 40 48
 *
 * Based on https://github.com/PascalPons/connect4
 *
 * Unlike the PascalPons package, we use 0-indexing for column indices.
 */
struct Game {
  struct Constants : public core::ConstantsBase {
    static constexpr const char* kGameName = "c4";
    using kNumActionsPerMode = util::int_sequence<kNumColumns>;
    static constexpr int kNumPlayers = 2;
    static constexpr int kMaxBranchingFactor = kNumColumns;
  };

  struct MctsConfiguration : public core::MctsConfigurationBase {
    static constexpr float kOpeningLength = 10.583;  // likely too big, just keeping previous value
  };

  struct State {
    auto operator<=>(const State& other) const = default;
    size_t hash() const;
    int num_empty_cells(column_t col) const;
    core::seat_index_t get_player_at(row_t row, column_t col) const;

    mask_t full_mask;        // spaces occupied by either player
    mask_t cur_player_mask;  // spaces occupied by current player
  };

  using GameResults = core::WinLossDrawResults;
  using StateHistory = core::SimpleStateHistory<State, Constants::kNumPreviousStatesToEncode>;
  using SymmetryGroup = groups::D1;
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;

  struct Symmetries {
    static Types::SymmetryMask get_mask(const State& state);
    static void apply(State& state, group::element_t sym);
    static void apply(StateHistory& history, group::element_t sym);  // optional
    static void apply(Types::PolicyTensor& policy, group::element_t sym, core::action_mode_t);
    static void apply(core::action_t& action, group::element_t sym, core::action_mode_t);
    static group::element_t get_canonical_symmetry(const State& state);
  };

  struct Rules : public game_base::RulesBase<Types> {
    static void init_state(State&);
    static Types::ActionMask get_legal_moves(const StateHistory&);
    static core::action_mode_t get_action_mode(const State&) { return 0; }
    static core::seat_index_t get_current_player(const State&);
    static void apply(StateHistory&, core::action_t action);
    static bool is_terminal(const State& state, core::seat_index_t last_player,
                            core::action_t last_action, GameResults::Tensor& outcome);
  };

  struct IO : public core::IOBase<Types> {
    static std::string action_delimiter() { return ""; }
    static std::string action_to_str(core::action_t action, core::action_mode_t) {
      return std::to_string(action + 1);
    }
    static std::string player_to_str(core::seat_index_t player);
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);

    static boost::json::value state_to_json(const State& state);

   private:
    static int print_row(char* buf, int n, const State&, row_t row, column_t blink_column);
  };

  static void static_init() {}

 private:
  static core::seat_index_t _get_player_at(const State& state, row_t row, column_t col);
  static constexpr int _to_bit_index(row_t row, column_t col);
  static constexpr mask_t _column_mask(column_t col);
  static constexpr mask_t _bottom_mask(column_t col);
  static constexpr mask_t _full_bottom_mask();
};

}  // namespace c4

namespace std {

template <>
struct hash<c4::Game::State> {
  size_t operator()(const c4::Game::State& pos) const { return pos.hash(); }
};

}  // namespace std

static_assert(core::concepts::Game<c4::Game>);

#include "inline/games/connect4/Game.inl"

// Add bindings at the end, after defining c4::Game. Adding here ensures that wherever we #include
// "games/connect4/Game.hpp", we also get the bindings.

#include "games/connect4/InputTensorizor.hpp"  // IWYU pragma: keep
#include "games/connect4/MctsEvalSpec.hpp"  // IWYU pragma: keep
#include "games/connect4/BayesianMctsEvalSpec.hpp"  // IWYU pragma: keep
