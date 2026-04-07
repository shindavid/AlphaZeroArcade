#pragma once

#include "core/BasicTypes.hpp"
#include "core/GameRulesBase.hpp"
#include "core/GameTypes.hpp"
#include "core/IOBase.hpp"
#include "core/WinLossPlayerResult.hpp"
#include "core/concepts/GameConcept.hpp"
#include "games/hex/Constants.hpp"
#include "games/hex/GameState.hpp"
#include "games/hex/Move.hpp"
#include "util/FiniteGroups.hpp"

#include <string>

namespace hex {

struct Game {
  using Constants = hex::Constants;

  using State = hex::GameState;
  using Move = hex::Move;
  using MoveSet = hex::MoveSet;
  using PlayerResult = core::WinLossPlayerResult;
  using SymmetryGroup = groups::C2;
  using Types = core::GameTypes<Constants, Move, MoveSet, State, PlayerResult, SymmetryGroup>;
  using GameOutcome = Types::GameOutcome;

  static constexpr int kVersion = 1;

  struct Rules : public core::RulesBase<Types> {
    static void init_state(State& s) { s.init(); }
    static core::seat_index_t get_current_player(const State& s) { return s.core.cur_player; }
    static void apply(State&, const Move&);
    static Result analyze(const State& state);

   private:
    static GameOutcome compute_outcome(const State& state);
  };

  struct IO : public core::IOBase<Types> {
    static constexpr char kSeatChars[Constants::kNumPlayers] = {'R', 'B'};
    static std::string action_delimiter() { return "-"; }
    static std::string player_to_str(core::seat_index_t player);
    static void print_state(std::ostream&, const State&, const Move* last_move = nullptr,
                            const Types::player_name_array_t* player_names = nullptr);

    static boost::json::value state_to_json(const State& state);
    static boost::json::value move_to_json_value(const Move& move) { return move.vertex(); }

   private:
    static int print_row(char* buf, int n, const State&, int row, int blink_column);
  };

  static void static_init() {}
};

}  // namespace hex

static_assert(core::concepts::Game<hex::Game>);

#include "inline/games/hex/Game.inl"
