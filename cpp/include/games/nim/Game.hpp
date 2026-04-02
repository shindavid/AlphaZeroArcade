#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "core/GameRulesBase.hpp"
#include "core/GameTypes.hpp"
#include "core/IOBase.hpp"
#include "core/WinShareResults.hpp"
#include "core/concepts/GameConcept.hpp"
#include "games/nim/Constants.hpp"
#include "games/nim/GameState.hpp"
#include "games/nim/Move.hpp"
#include "util/FiniteGroups.hpp"

#include <boost/functional/hash.hpp>

#include <string>

namespace nim {

struct Game {
  struct Constants : public core::ConstantsBase {
    static constexpr const char* kGameName = "nim";
    using kNumActionsPerMode = util::int_sequence<nim::kMaxStonesToTake>;
    static constexpr int kNumPlayers = nim::kNumPlayers;
    static constexpr int kNumMoves = nim::kMaxStonesToTake;
    static constexpr int kMaxBranchingFactor = nim::kMaxStonesToTake;
  };

  using State = nim::GameState;
  using Move = nim::Move;
  using MoveList = nim::MoveList;
  using GameResults = core::WinShareResults<Constants::kNumPlayers>;
  using SymmetryGroup = groups::TrivialGroup;
  using Types = core::GameTypes<Constants, Move, MoveList, State, GameResults, SymmetryGroup>;

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

    static int move_to_json_value(const Move& move) { return int(move); }
    static std::string move_to_str(const Move& move) { return std::to_string(int(move) + 1); }
    static Move move_from_str(const GameState&, std::string_view s) { return Move(util::atoi(s) - 1); }
    static std::string serialize_move(const Move& move) { return std::format("{}", int(move)); }
    static Move deserialize_move(std::string_view s) { return Move(util::atoi(s) - 1); }
  };

  static void static_init() {}
};  // struct Game

}  // namespace nim

static_assert(core::concepts::Game<nim::Game>);

template <>
struct std::formatter<nim::Move> : std::formatter<std::string> {
  auto format(const nim::Move& move, format_context& ctx) const {
    return std::formatter<std::string>::format(nim::Game::IO::move_to_str(move), ctx);
  }
};

#include "inline/games/nim/Game.inl"
//
// Ensure that we always have bindings when we #include "games/nim/Game.hpp":
#include "games/nim/Bindings.hpp"  // IWYU pragma: keep
