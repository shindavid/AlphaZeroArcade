#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "core/GameRulesBase.hpp"
#include "core/GameTypes.hpp"
#include "core/IOBase.hpp"
#include "core/WinLossDrawResults.hpp"
#include "core/concepts/GameConcept.hpp"
#include "games/othello/Constants.hpp"
#include "games/othello/GameState.hpp"
#include "games/othello/Move.hpp"
#include "util/FiniteGroups.hpp"

#include <boost/functional/hash.hpp>

#include <cstdint>
#include <string>

namespace othello {

/*
 * See <othello/Constants.hpp> for bitboard representation details.
 *
 * The algorithms for manipulating the board are adapted from:
 *
 * https://github.com/abulmo/edax-reversi
 */
class Game {
 public:
  struct Constants : public core::ConstantsBase {
    static constexpr const char* kGameName = "othello";
    static constexpr int kNumPlayers = 2;
    static constexpr int kNumMoves = kNumGlobalActions;
    static constexpr int kMaxBranchingFactor = othello::kMaxNumLocalActions;
  };

  using State = othello::GameState;
  using Move = othello::Move;
  using MoveList = othello::MoveList;
  using GameResults = core::WinLossDrawResults;
  using SymmetryGroup = groups::D4;
  using Types = core::GameTypes<Constants, Move, MoveList, State, GameResults, SymmetryGroup>;

  struct Rules : public core::RulesBase<Types> {
    static void init_state(State&);
    static core::seat_index_t get_current_player(const State&);
    static void apply(State&, const Move& move);
    static Result analyze(const State& state);

   private:
    static GameResults::Tensor compute_outcome(const State& state);
  };

  struct IO : public core::IOBase<Types> {
    static constexpr char kSeatChars[kNumPlayers] = {'B', 'W'};
    static std::string action_delimiter() { return "-"; }
    static std::string player_to_str(core::seat_index_t player);
    static void print_state(std::ostream&, const State&, const Move* last_move = nullptr,
                            const Types::player_name_array_t* player_names = nullptr);

    static void write_edax_board_str(char* buf, const State& state);
    static boost::json::value state_to_json(const State& state);

    static int move_to_json_value(const Move& move) { return int(move); }
    static std::string move_to_str(const Move&);
    static Move move_from_str(const GameState&, std::string_view s);
    static std::string serialize_move(const Move&);
    static Move deserialize_move(std::string_view s);

   private:
    static int print_row(char* buf, int n, const State&, const MoveList&, row_t row,
                         column_t blink_column);
  };

  static void static_init() {}

 private:
  static mask_t get_moves(mask_t P, mask_t O);
  static mask_t get_some_moves(mask_t P, mask_t mask, int dir);
};

extern uint64_t (*flip[kNumGlobalActions])(const uint64_t, const uint64_t);

}  // namespace othello

static_assert(core::concepts::Game<othello::Game>);

template <>
struct std::formatter<othello::Move> : std::formatter<std::string> {
  auto format(const othello::Move& move, format_context& ctx) const {
    return std::formatter<std::string>::format(othello::Game::IO::move_to_str(move), ctx);
  }
};

#include "inline/games/othello/Game.inl"

// Ensure that we always have bindings when we #include "games/othello/Game.hpp":
#include "games/othello/Bindings.hpp"  // IWYU pragma: keep
