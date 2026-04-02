#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "core/GameRulesBase.hpp"
#include "core/GameTypes.hpp"
#include "core/IOBase.hpp"
#include "core/WinLossDrawResults.hpp"
#include "core/concepts/GameConcept.hpp"
#include "games/chess/Constants.hpp"
#include "games/chess/GameState.hpp"
#include "games/chess/InputFrame.hpp"
#include "games/chess/Move.hpp"
#include "util/FiniteGroups.hpp"

#include <boost/functional/hash.hpp>

#include <string>

namespace a0achess {

struct Game {
  struct Constants : public core::ConstantsBase {
    static constexpr const char* kGameName = "chess";
    static constexpr int kNumPlayers = a0achess::kNumPlayers;
    static constexpr int kNumMoves = a0achess::kNumMoves;
    static constexpr int kMaxBranchingFactor = a0achess::kMaxBranchingFactor;

    // 2 = official rules, 1 = engine rules
    static constexpr int kRepetitionDrawThreshold = 1;
  };

  using State = GameState;
  using Move = a0achess::Move;
  using MoveList = a0achess::MoveList;
  using GameResults = core::WinLossDrawResults;
  using SymmetryGroup = groups::TrivialGroup;  // TODO: Implement symmetries
  using Types = core::GameTypes<Constants, Move, MoveList, State, GameResults, SymmetryGroup>;

  struct Rules : public core::RulesBase<Types> {
    static void init_state(State&);
    static core::game_phase_t get_game_phase(const State&);
    static core::seat_index_t get_current_player(const State&);
    static void apply(State& state, const Move& move) { state.makeMove(move); }
    static void backtrack_state(State& state, const State& prev) { state.backtrack_to(prev); }
    static Result analyze(const State& state);
    static Result analyze(const InputFrame&);
  };

  struct IO : public core::IOBase<Types> {
    static constexpr char kSeatChars[Constants::kNumPlayers] = {'W', 'B'};
    static std::string action_delimiter() { return "-"; }
    static void print_state(std::ostream&, const State&, const Move* last_move = nullptr,
                            const Types::player_name_array_t* player_names = nullptr);
  };

  static void static_init() {}
};

}  // namespace a0achess

static_assert(core::concepts::Game<a0achess::Game>);

#include "inline/games/chess/Game.inl"

// Ensure that we always have bindings when we #include "games/chess/Game.hpp":
#include "games/chess/Bindings.hpp"  // IWYU pragma: keep
