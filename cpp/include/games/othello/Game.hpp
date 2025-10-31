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
#include "util/FiniteGroups.hpp"

#include <boost/functional/hash.hpp>

#include <cstdint>
#include <functional>
#include <string>

namespace othello {

/*
 * See <othello/Constants.hpp> for bitboard representation details.
 *
 * The algorithms for manipulating the board are lifted from:
 *
 * https://github.com/abulmo/edax-reversi
 */
class Game {
 public:
  struct Constants : public core::ConstantsBase {
    static constexpr const char* kGameName = "othello";
    using kNumActionsPerMode = util::int_sequence<othello::kNumGlobalActions>;
    static constexpr int kNumPlayers = 2;
    static constexpr int kMaxBranchingFactor = othello::kMaxNumLocalActions;
  };

  using State = othello::GameState;
  using GameResults = core::WinLossDrawResults;
  using SymmetryGroup = groups::D4;
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;

  struct Symmetries {
    static Types::SymmetryMask get_mask(const State& state);
    static void apply(State& state, group::element_t sym);
    template <eigen_util::concepts::FTensor Tensor>
    static void apply(Tensor& tensor, group::element_t sym, core::action_mode_t = 0);
    static void apply(core::action_t& action, group::element_t sym, core::action_mode_t = 0);
    static group::element_t get_canonical_symmetry(const State& state);
  };

  struct Rules : public core::RulesBase<Types> {
    static void init_state(State&);
    static Types::ActionMask get_legal_moves(const State&);
    static core::action_mode_t get_action_mode(const State&) { return 0; }
    static core::seat_index_t get_current_player(const State&);
    static void apply(State&, core::action_t action);
    static bool is_terminal(const State& state, core::seat_index_t last_player,
                            core::action_t last_action, GameResults::Tensor& outcome);

   private:
    static GameResults::Tensor compute_outcome(const State& state);
  };

  struct IO : public core::IOBase<Types> {
    static constexpr char kSeatChars[kNumPlayers] = {'B', 'W'};
    static std::string action_delimiter() { return "-"; }
    static std::string action_to_str(core::action_t action, core::action_mode_t);
    static std::string player_to_str(core::seat_index_t player);
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);

    static void write_edax_board_str(char* buf, const State& state);
    static int seat_char_to_index(char seat_char);

    static boost::json::value state_to_json(const State& state);

   private:
    static int print_row(char* buf, int n, const State&, const Types::ActionMask&, row_t row,
                         column_t blink_column);
  };

  static void static_init() {}

 private:
  static mask_t get_moves(mask_t P, mask_t O);
  static mask_t get_some_moves(mask_t P, mask_t mask, int dir);
};

extern uint64_t (*flip[kNumGlobalActions])(const uint64_t, const uint64_t);

}  // namespace othello

namespace std {

template <>
struct hash<othello::Game::State> {
  size_t operator()(const othello::Game::State& pos) const { return pos.hash(); }
};

}  // namespace std

static_assert(core::concepts::Game<othello::Game>);

#include "inline/games/othello/Game.inl"

// Ensure that we always have bindings when we #include "games/othello/Game.hpp":
#include "games/othello/Bindings.hpp"  // IWYU pragma: keep
