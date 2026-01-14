#pragma once

#include "core/BasicTypes.hpp"
#include "core/GameRulesBase.hpp"
#include "core/GameTypes.hpp"
#include "core/IOBase.hpp"
#include "core/WinLossResults.hpp"
#include "core/concepts/GameConcept.hpp"
#include "games/hex/Constants.hpp"
#include "games/hex/GameState.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

#include <string>

namespace hex {

struct Game {
  using Constants = hex::Constants;

  using State = hex::GameState;
  using GameResults = core::WinLossResults;
  using SymmetryGroup = groups::C2;
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;

  struct Symmetries {
    static Types::SymmetryMask get_mask(const State&) { return Types::SymmetryMask().set(); }
    static void apply(State& state, group::element_t sym);
    template <eigen_util::concepts::FTensor Tensor>
    static void apply(Tensor& tensor, group::element_t sym, core::action_mode_t);
    static void apply(core::action_t& action, group::element_t sym, core::action_mode_t);
  };

  struct Rules : public core::RulesBase<Types> {
    static void init_state(State& s) { s.init(); }
    static Types::ActionMask get_legal_moves(const State&);
    static core::action_mode_t get_action_mode(const State&) { return 0; }
    static core::seat_index_t get_current_player(const State& s) { return s.core.cur_player; }
    static void apply(State&, core::action_t action);
    static bool is_terminal(const State& state, core::seat_index_t last_player,
                            core::action_t last_action, GameResults::Tensor& outcome);

   private:
    static core::action_t compute_mirror_action(core::action_t action);
    static vertex_t to_vertex(int row, int col) { return row * Constants::kBoardDim + col; }
    static GameResults::Tensor compute_outcome(const State& state);
  };

  struct IO : public core::IOBase<Types> {
    static constexpr char kSeatChars[Constants::kNumPlayers] = {'R', 'B'};
    static std::string action_delimiter() { return "-"; }
    static std::string action_to_str(core::action_t action, core::action_mode_t mode = 0);
    static std::string player_to_str(core::seat_index_t player);
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);

    static boost::json::value state_to_json(const State& state);

   private:
    static int print_row(char* buf, int n, const State&, int row, int blink_column);
  };

  static void static_init() {}
};

}  // namespace hex

static_assert(core::concepts::Game<hex::Game>);

#include "inline/games/hex/Game.inl"

// IWYU pragma: keep
// Ensure that we always have bindings when we #include "games/hex/Game.hpp":
#include "games/hex/Bindings.hpp"  // IWYU pragma: keep
