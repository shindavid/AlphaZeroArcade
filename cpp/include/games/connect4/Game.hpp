#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "core/GameRulesBase.hpp"
#include "core/GameTypes.hpp"
#include "core/IOBase.hpp"
#include "core/WinLossDrawResults.hpp"
#include "core/concepts/GameConcept.hpp"
#include "games/connect4/Constants.hpp"
#include "games/connect4/GameState.hpp"
#include "util/CppUtil.hpp"
#include "util/FiniteGroups.hpp"

#include <boost/json.hpp>

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

  using State = GameState;
  using GameResults = core::WinLossDrawResults;
  using SymmetryGroup = groups::D1;
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;

  struct Rules : public core::RulesBase<Types> {
    static void init_state(State& state) { state.init(); }
    static core::action_mode_t get_action_mode(const State&) { return 0; }
    static core::seat_index_t get_current_player(const State&);
    static void apply(State&, core::action_t action);
    static Result analyze(const State& state);

   private:
    static Types::ActionMask get_legal_moves(const State& state);
  };

  struct IO : public core::IOBase<Types> {
    static constexpr char kSeatChars[Constants::kNumPlayers] = {'R', 'Y'};
    static std::string action_delimiter() { return ""; }
    static std::string action_to_str(core::action_t action, core::action_mode_t) {
      return std::to_string(action + 1);
    }
    static std::string player_to_str(core::seat_index_t player);
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);

    static boost::json::value state_to_json(const State& state);
    static void add_render_info(const State& state, boost::json::object& obj);

   private:
    static int print_row(char* buf, int n, const State&, row_t row, column_t blink_column);
  };

  static void static_init() {}
};

}  // namespace c4

static_assert(core::concepts::Game<c4::Game>);

#include "inline/games/connect4/Game.inl"

// Ensure that we always have bindings when we #include "games/connect4/Game.hpp":
#include "games/connect4/Bindings.hpp"  // IWYU pragma: keep
