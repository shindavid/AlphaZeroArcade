#pragma once

#include "core/BasicTypes.hpp"
#include "core/GameTypes.hpp"
#include "core/IOBase.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/SimpleStateHistory.hpp"
#include "core/TrainingTargets.hpp"
#include "core/WinLossResults.hpp"
#include "core/concepts/Game.hpp"
#include "games/GameRulesBase.hpp"
#include "games/hex/Constants.hpp"
#include "games/hex/GameState.hpp"
#include "util/FiniteGroups.hpp"
#include "util/MetaProgramming.hpp"

#include <string>

namespace hex {

struct Game {
  using Constants = hex::Constants;

  struct MctsConfiguration : public core::MctsConfigurationBase {
    static constexpr float kOpeningLength = 8;
  };

  using State = hex::GameState;
  using GameResults = core::WinLossResults;
  using StateHistory = core::SimpleStateHistory<State, Constants::kNumPreviousStatesToEncode>;
  using SymmetryGroup = groups::C2;
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;

  struct Symmetries {
    static Types::SymmetryMask get_mask(const State&) { return Types::SymmetryMask().set(); }
    static void apply(State& state, group::element_t sym);
    static void apply(StateHistory& history, group::element_t sym);  // optional
    static void apply(Types::PolicyTensor& policy, group::element_t sym, core::action_mode_t);
    static void apply(core::action_t& action, group::element_t sym, core::action_mode_t);
    static group::element_t get_canonical_symmetry(const State& state);
  };

  struct Rules : public game_base::RulesBase<Types> {
    static void init_state(State& s) { s.init(); }
    static Types::ActionMask get_legal_moves(const StateHistory&);
    static core::action_mode_t get_action_mode(const State&) { return 0; }
    static core::seat_index_t get_current_player(const State& s) { return s.core.cur_player; }
    static void apply(StateHistory&, core::action_t action);
    static bool is_terminal(const State& state, core::seat_index_t last_player,
                            core::action_t last_action, GameResults::Tensor& outcome);
    static Types::ActionMask get_legal_moves(const State&);

   private:
    static core::action_t compute_mirror_action(core::action_t action);
    static vertex_t to_vertex(int row, int col) { return row * Constants::kBoardDim + col; }
    static GameResults::Tensor compute_outcome(const State& state);
  };

  struct IO : public core::IOBase<Types> {
    static std::string action_delimiter() { return "-"; }
    static std::string action_to_str(core::action_t action, core::action_mode_t mode = 0);
    static std::string player_to_str(core::seat_index_t player);
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);

   private:
    static int print_row(char* buf, int n, const State&, int row, int blink_column);
  };

  struct InputTensorizor {
    // TODO: add more feature planes

    // +1 for swap-legality plane
    static constexpr int kDim0 =
      1 + Constants::kNumPlayers * (1 + Constants::kNumPreviousStatesToEncode);

    using Shape = Eigen::Sizes<kDim0, Constants::kBoardDim, Constants::kBoardDim>;
    using Tensor = eigen_util::FTensor<Shape>;
    using MCTSKey = State;
    using EvalKey = State;

    static MCTSKey mcts_key(const StateHistory& history) { return history.current(); }
    template <typename Iter>
    static EvalKey eval_key(Iter start, Iter cur) {
      return *cur;
    }
    template <typename Iter>
    static Tensor tensorize(Iter start, Iter cur);
  };

  struct TrainingTargets {
    using BoardShape = Eigen::Sizes<Constants::kBoardDim, Constants::kBoardDim>;

    using PolicyTarget = core::PolicyTarget<Game>;
    using ValueTarget = core::ValueTarget<Game>;
    using ActionValueTarget = core::ActionValueTarget<Game>;
    using OppPolicyTarget = core::OppPolicyTarget<Game>;

    using List = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget, OppPolicyTarget>;
  };

  static void static_init() {}
};

}  // namespace hex

static_assert(core::concepts::Game<hex::Game>);

#include "inline/games/hex/Game.inl"
