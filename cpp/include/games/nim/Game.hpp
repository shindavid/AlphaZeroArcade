#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "core/GameRulesBase.hpp"
#include "core/GameTraits.hpp"
#include "core/IOBase.hpp"
#include "core/WinSharePlayerResult.hpp"
#include "core/concepts/GameConcept.hpp"
#include "games/nim/Constants.hpp"
#include "games/nim/GameState.hpp"
#include "games/nim/Move.hpp"
#include "util/FiniteGroups.hpp"

#include <boost/functional/hash.hpp>

#include <string>

namespace nim {

struct Game {
  static constexpr int kVersion = 1;

  struct Constants : public core::ConstantsBase {
    static constexpr const char* kGameName = "nim";
    using kNumActionsPerMode = util::int_sequence<nim::kMaxStonesToTake>;
    static constexpr int kNumPlayers = nim::kNumPlayers;
    static constexpr int kNumMoves = nim::kMaxStonesToTake;
    static constexpr int kMaxBranchingFactor = nim::kMaxStonesToTake;
  };

  using State = nim::GameState;
  using Move = nim::Move;
  using MoveSet = nim::MoveSet;
  using PlayerResult = core::WinSharePlayerResult;
  using SymmetryGroup = groups::TrivialGroup;
  using Types = core::GameTraits<Constants, Move, MoveSet, State, PlayerResult, SymmetryGroup>;

  struct Rules : public core::RulesBase<Types> {
    static void init_state(State& state);
    static core::seat_index_t get_current_player(const State& state);
    static void apply(State& state, const Move& move);
    static Result analyze(const State& state);
  };

  struct IO : public core::IOBase<Types> {
    static constexpr char kSeatChars[Constants::kNumPlayers] = {'A', 'B'};
    static std::string action_delimiter() { return "-"; }
    static void print_state(std::ostream&, const State&, const Move* last_move = nullptr,
                            const Types::player_name_array_t* player_names = nullptr);
    static std::string compact_state_repr(const State& state);
    static boost::json::value move_to_json_value(const Move& move) { return int(move); }
  };
};  // struct Game

}  // namespace nim

static_assert(core::concepts::Game<nim::Game>);

#include "inline/games/nim/Game.inl"
