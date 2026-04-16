#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "core/GameRulesBase.hpp"
#include "core/GameTraits.hpp"
#include "core/IOBase.hpp"
#include "core/TrivialSymmetries.hpp"
#include "core/WinSharePlayerResult.hpp"
#include "core/concepts/GameConcept.hpp"
#include "games/stochastic_nim/ChanceDistribution.hpp"
#include "games/stochastic_nim/Constants.hpp"
#include "games/stochastic_nim/GameState.hpp"
#include "games/stochastic_nim/Move.hpp"
#include "util/FiniteGroups.hpp"

#include <boost/functional/hash.hpp>

#include <string>

namespace stochastic_nim {

struct Game {
  static constexpr int kVersion = 1;

  struct Constants : public core::ConstantsBase {
    static constexpr const char* kGameName = "stochastic_nim";
    static constexpr int kNumPlayers = stochastic_nim::kNumPlayers;
    static constexpr int kNumMoves = stochastic_nim::kNumMoves;
    static constexpr int kMaxBranchingFactor = stochastic_nim::kNumMoves;
  };

  using State = stochastic_nim::GameState;
  using InfoSet = State;
  using Move = stochastic_nim::Move;
  using MoveSet = stochastic_nim::MoveSet;
  using ChanceDistribution = stochastic_nim::ChanceDistribution;
  using PlayerResult = core::WinSharePlayerResult;
  using SymmetryGroup = groups::TrivialGroup;
  using Symmetries = core::TrivialSymmetries;
  using Types = core::GameTraits<Constants, Move, MoveSet, State, InfoSet, PlayerResult, SymmetryGroup>;

  struct Rules : public core::RulesBase<Types> {
    static void init_state(State& state);
    static core::seat_index_t get_current_player(const State& state);
    static void apply(State& state, const Move& move);
    static bool is_chance_state(const State&);
    static ChanceDistribution get_chance_distribution(const State& state);
    static Result analyze(const State& state);

   private:
    static MoveSet get_legal_moves(const State& state);
  };

  struct IO : public core::IOBase<Types> {
    static constexpr char kSeatChars[Constants::kNumPlayers] = {'A', 'B'};
    static std::string action_delimiter() { return "-"; }
    static void print_state(std::ostream& ss, const State& state, const Move* last_move = nullptr,
                            const Types::player_name_array_t* player_names = nullptr);
    static std::string compact_state_repr(const State& state);
    static boost::json::value move_to_json_value(const Move& move) { return move.index(); }
  };
};  // struct Game
}  // namespace stochastic_nim

static_assert(core::concepts::Game<stochastic_nim::Game>);

#include "inline/games/stochastic_nim/Game.inl"
