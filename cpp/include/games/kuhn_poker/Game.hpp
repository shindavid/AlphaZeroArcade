#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "core/GameRulesBase.hpp"
#include "core/GameTraits.hpp"
#include "core/IOBase.hpp"
#include "core/ScorePlayerResult.hpp"
#include "core/concepts/GameConcept.hpp"
#include "games/kuhn_poker/ChanceDistribution.hpp"
#include "games/kuhn_poker/Constants.hpp"
#include "games/kuhn_poker/GameState.hpp"
#include "games/kuhn_poker/InfoSetState.hpp"
#include "games/kuhn_poker/Move.hpp"
#include "util/FiniteGroups.hpp"

#include <string>

namespace kuhn_poker {

struct Game {
  static constexpr int kVersion = 1;

  struct Constants : public core::ConstantsBase {
    static constexpr const char* kGameName = "kuhn_poker";
    static constexpr int kNumPlayers = kuhn_poker::kNumPlayers;
    static constexpr int kMaxBranchingFactor = kuhn_poker::kNumDeals;
  };

  using State = kuhn_poker::GameState;
  using InfoSet = kuhn_poker::InfoSetState;
  using Move = kuhn_poker::Move;
  using MoveSet = kuhn_poker::MoveSet;
  using ChanceDistribution = kuhn_poker::ChanceDistribution;
  using PlayerResult = core::ScorePlayerResult;
  using SymmetryGroup = groups::TrivialGroup;
  using Types = core::GameTraits<Constants, Move, MoveSet, State, InfoSet, PlayerResult, SymmetryGroup>;

  struct Rules : public core::RulesBase<Types> {
    static void init_state(State& state);
    static core::seat_index_t get_current_player(const State& state);
    static void apply(State& state, const Move& move);
    static bool is_chance_state(const State& state);
    static ChanceDistribution get_chance_distribution(const State& state);
    static Result analyze(const State& state);
    static InfoSet state_to_info_set(const State& state, core::seat_index_t seat);
  };

  struct IO : public core::IOBase<Types> {
    static constexpr char kSeatChars[Constants::kNumPlayers] = {'0', '1'};
    static std::string action_delimiter() { return "-"; }
    static std::string player_to_str(core::seat_index_t player);
    static void print_state(std::ostream&, const State&, const Move* last_move = nullptr,
                            const Types::player_name_array_t* player_names = nullptr);
    static std::string compact_state_repr(const State& state);
    static boost::json::value move_to_json_value(const Move& move) { return move.index(); }
  };
};  // struct Game

}  // namespace kuhn_poker

static_assert(core::concepts::Game<kuhn_poker::Game>);

#include "inline/games/kuhn_poker/Game.inl"
