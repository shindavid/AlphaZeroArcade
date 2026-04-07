#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "core/GameRulesBase.hpp"
#include "core/GameTypes.hpp"
#include "core/IOBase.hpp"
#include "core/WinLossDrawPlayerResult.hpp"
#include "core/concepts/GameConcept.hpp"
#include "games/tictactoe/Constants.hpp"
#include "games/tictactoe/GameState.hpp"
#include "games/tictactoe/Move.hpp"
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
  static constexpr int kVersion = 1;
  struct Constants : public core::ConstantsBase {
    static constexpr const char* kGameName = "tictactoe";
    static constexpr int kNumPlayers = tictactoe::kNumPlayers;
    static constexpr int kNumMoves = tictactoe::kNumCells;
    static constexpr int kMaxBranchingFactor = tictactoe::kNumCells;
  };

  using State = GameState;
  using Move = tictactoe::Move;
  using MoveSet = tictactoe::MoveSet;
  using PlayerResult = core::WinLossDrawPlayerResult;
  using SymmetryGroup = groups::D4;
  using Types = core::GameTypes<Constants, Move, MoveSet, State, PlayerResult, SymmetryGroup>;
  using GameOutcome = Types::GameOutcome;

  struct Rules : public core::RulesBase<Types> {
    static void init_state(State&);
    static core::seat_index_t get_current_player(const State&);
    static void apply(State&, const Move&);
    static Result analyze(const State& state);

   private:
    static MoveSet get_legal_moves(const State& state);
  };

  struct IO : public core::IOBase<Types> {
    static constexpr char kSeatChars[Constants::kNumPlayers] = {'X', 'O'};
    static std::string action_delimiter() { return ""; }
    static std::string player_to_str(core::seat_index_t player) {
      return (player == tictactoe::kX) ? "X" : "O";
    }
    static void print_state(std::ostream&, const State&, const Move* last_move = nullptr,
                            const Types::player_name_array_t* player_names = nullptr);
    static std::string compact_state_repr(const State& state);
    static boost::json::value state_to_json(const State& state);
    static boost::json::value move_to_json_value(const Move& move) { return int(move); }
  };

  static constexpr mask_t kThreeInARowMasks[] = {
    make_mask(0, 1, 2), make_mask(3, 4, 5), make_mask(6, 7, 8), make_mask(0, 3, 6),
    make_mask(1, 4, 7), make_mask(2, 5, 8), make_mask(0, 4, 8), make_mask(2, 4, 6)};

  static void static_init() {}
};

}  // namespace tictactoe

static_assert(core::concepts::Game<tictactoe::Game>);

#include "inline/games/tictactoe/Game.inl"
