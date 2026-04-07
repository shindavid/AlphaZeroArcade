#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "core/GameRulesBase.hpp"
#include "core/GameTypes.hpp"
#include "core/IOBase.hpp"
#include "core/WinSharePlayerResult.hpp"
#include "core/concepts/GameConcept.hpp"
#include "games/blokus/Constants.hpp"
#include "games/blokus/GameState.hpp"
#include "games/blokus/Move.hpp"
#include "util/FiniteGroups.hpp"

#include <boost/functional/hash.hpp>

#include <string>

namespace blokus {

class Game {
 public:
  static constexpr int kVersion = 1;

  struct Constants : public core::ConstantsBase {
    static constexpr const char* kGameName = "blokus";
    static constexpr int kNumPlayers = blokus::kNumPlayers;
    static constexpr int kNumMoves = blokus::kNumMoves;
    static constexpr int kMaxBranchingFactor = blokus::kNumPieceOrientationCorners;
  };

  using State = blokus::GameState;
  using Move = blokus::Move;
  using MoveSet = blokus::MoveSet;
  using PlayerResult = core::WinSharePlayerResult;

  /*
   * After the initial placement of the first piece, the rules of the game are symmetric. But the
   * rules are not symmetric for the first piece placement, and as a result, strategic
   * considerations are asymmetric for much, if not all of the game. Because of this, it's unclear
   * whether exploiting symmetry will be useful, so we use the trivial group.
   */
  using SymmetryGroup = groups::TrivialGroup;
  using Types = core::GameTypes<Constants, Move, MoveSet, State, PlayerResult, SymmetryGroup>;
  using GameOutcome = Types::GameOutcome;

  struct Rules : public core::RulesBase<Types> {
    static void init_state(State&);
    static core::seat_index_t get_current_player(const State& s) { return s.core.cur_color; }
    static void apply(State&, const Move&);
    static Result analyze(const State& state);

   private:
    static GameOutcome compute_outcome(const State& state);
    static MoveSet get_legal_moves(const State& state);
  };

  struct IO : public core::IOBase<Types> {
    static constexpr char kSeatChars[Constants::kNumPlayers] = {'B', 'Y', 'R', 'G'};
    static std::string action_delimiter() { return "-"; }
    static std::string player_to_str(core::seat_index_t player);
    static void print_state(std::ostream&, const State&, const Move* last_move = nullptr,
                            const Types::player_name_array_t* player_names = nullptr);

    /*
     * Inverse operation of print_state(ss, state) in non-tty-mode.
     *
     * Assumes that the last pass_count players have passed.
     */
    static State load(const std::string& str, int pass_count = 0);
    static boost::json::value move_to_json_value(const Move& move) { return move.index(); }
  };

  static void static_init() {}
};

}  // namespace blokus

static_assert(core::concepts::Game<blokus::Game>);
