#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "core/GameRulesBase.hpp"
#include "core/GameTraits.hpp"
#include "core/IOBase.hpp"
#include "core/WinLossDrawPlayerResult.hpp"
#include "core/concepts/GameConcept.hpp"
#include "games/connect4/Constants.hpp"
#include "games/connect4/GameState.hpp"
#include "games/connect4/Move.hpp"
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
  static constexpr int kVersion = 1;

  struct Constants : public core::ConstantsBase {
    static constexpr const char* kGameName = "c4";
    static constexpr int kNumPlayers = 2;
    static constexpr int kNumMoves = kNumColumns;
    static constexpr int kMaxBranchingFactor = kNumColumns;
  };

  using State = GameState;
  using InfoSet = State;
  using Move = c4::Move;
  using MoveSet = c4::MoveSet;
  using PlayerResult = core::WinLossDrawPlayerResult;
  using SymmetryGroup = groups::D1;
  using Types =
    core::GameTraits<Constants, Move, MoveSet, State, InfoSet, PlayerResult, SymmetryGroup>;

  struct Rules : public core::RulesBase<Types> {
    static void init_state(State& state) { state.init(); }
    static core::seat_index_t get_current_player(const State&);
    static void apply(State&, const Move& move);
    static Result analyze(const State& state);

   private:
    static MoveSet get_legal_moves(const State& state);
  };

  struct IO : public core::IOBase<Types> {
    static constexpr char kSeatChars[Constants::kNumPlayers] = {'R', 'Y'};
    static std::string action_delimiter() { return ""; }
    static std::string player_to_str(core::seat_index_t player);
    static std::string compact_state_repr(const State& state);
    static void print_state(std::ostream&, const State&, const Move* last_move = nullptr,
                            const Types::player_name_array_t* player_names = nullptr);

    static boost::json::value info_set_to_json(const InfoSet& info_set);
    static void add_render_info(const InfoSet& info_set, boost::json::object& obj);
    static boost::json::value move_to_json_value(const Move& move) { return int(move); }

   private:
    static int print_row(char* buf, int n, const State&, row_t row, column_t blink_column);
  };
};

}  // namespace c4

static_assert(core::concepts::Game<c4::Game>);

#include "inline/games/connect4/Game.inl"
