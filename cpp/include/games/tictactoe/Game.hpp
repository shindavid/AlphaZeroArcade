#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <core/ConstantsBase.hpp>
#include <core/GameLog.hpp>
#include <core/GameTypes.hpp>
#include <core/IOBase.hpp>
#include <core/MctsConfigurationBase.hpp>
#include <core/SimpleStateHistory.hpp>
#include <core/TrainingTargets.hpp>
#include <core/WinLossDrawResults.hpp>
#include <games/tictactoe/Constants.hpp>
#include <games/GameRulesBase.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>
#include <util/MetaProgramming.hpp>

#include <boost/functional/hash.hpp>
#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>

namespace tictactoe {

constexpr mask_t make_mask(int a, int b, int c) {
  return (mask_t(1) << a) + (mask_t(1) << b) + (mask_t(1) << c);
}

/*
 * Bit order encoding for the board:
 *
 * 0 1 2
 * 3 4 5
 * 6 7 8
 */
class Game {
 public:
  struct Constants : public core::ConstantsBase {
    using kNumActionsPerMode = util::int_sequence<tictactoe::kNumCells>;
    static constexpr int kNumPlayers = tictactoe::kNumPlayers;
    static constexpr int kMaxBranchingFactor = tictactoe::kNumCells;
  };

  struct MctsConfiguration : public core::MctsConfigurationBase {
    static constexpr float kOpeningLength = 4;
  };

  struct State {
    auto operator<=>(const State& other) const = default;
    size_t hash() const;
    mask_t opponent_mask() const { return full_mask ^ cur_player_mask; }

    mask_t full_mask;        // spaces occupied by either player
    mask_t cur_player_mask;  // spaces occupied by current player
  };

  using GameResults = core::WinLossDrawResults;
  using StateHistory = core::SimpleStateHistory<State, Constants::kNumPreviousStatesToEncode>;
  using SymmetryGroup = groups::D4;
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;

  struct Symmetries {
    static Types::SymmetryMask get_mask(const State& state);
    static void apply(State& state, group::element_t sym);
    static void apply(StateHistory& history, group::element_t sym);  // optional
    static void apply(Types::PolicyTensor& policy, group::element_t sym, core::action_mode_t=0);
    static void apply(core::action_t& action, group::element_t sym, core::action_mode_t=0);
    static group::element_t get_canonical_symmetry(const State& state);
  };

  struct Rules : public game_base::RulesBase<Types> {
    static void init_state(State&);
    static Types::ActionMask get_legal_moves(const StateHistory&);
    static core::action_mode_t get_action_mode(const State&) { return 0; }
    static core::seat_index_t get_current_player(const State&);
    static void apply(StateHistory&, core::action_t action);
    static bool is_terminal(const State& state, core::seat_index_t last_player,
                            core::action_t last_action, GameResults::Tensor& outcome);
  };

  struct IO : public core::IOBase<Types, State> {
    static std::string action_delimiter() { return ""; }
    static std::string action_to_str(core::action_t action, core::action_mode_t) {
      return std::to_string(action);
    }
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);
    static void print_mcts_results(std::ostream&, const Types::PolicyTensor& action_policy,
                                   const Types::SearchResults&);
    static std::string compact_state_repr(const State& state);
  };

  struct InputTensorizor {
    static constexpr int kDim0 = kNumPlayers * (1 + Constants::kNumPreviousStatesToEncode);
    using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDimension, kBoardDimension>>;
    using MCTSKey = State;
    using EvalKey = State;

    static MCTSKey mcts_key(const StateHistory& history) { return history.current(); }
    template <typename Iter> static EvalKey eval_key(Iter start, Iter cur) { return *cur; }
    template <typename Iter> static Tensor tensorize(Iter start, Iter cur);
  };

  struct TrainingTargets {
    using BoardShape = Eigen::Sizes<kBoardDimension, kBoardDimension>;
    using OwnershipShape = Eigen::Sizes<3, kBoardDimension, kBoardDimension>;

    using PolicyTarget = core::PolicyTarget<Game>;
    using ValueTarget = core::ValueTarget<Game>;
    using ActionValueTarget = core::ActionValueTarget<Game>;
    using OppPolicyTarget = core::OppPolicyTarget<Game>;

    struct OwnershipTarget {
      static constexpr const char* kName = "ownership";
      using Tensor = eigen_util::FTensor<OwnershipShape>;

      static Tensor tensorize(const Types::GameLogView& view);
    };

    using List = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget, OppPolicyTarget,
                              OwnershipTarget>;
  };

  static constexpr mask_t kThreeInARowMasks[] = {
      make_mask(0, 1, 2), make_mask(3, 4, 5), make_mask(6, 7, 8), make_mask(0, 3, 6),
      make_mask(1, 4, 7), make_mask(2, 5, 8), make_mask(0, 4, 8), make_mask(2, 4, 6)};

  static void static_init() {}

 private:
  static core::seat_index_t _get_player_at(const State& state, int row, int col);
};

}  // namespace tictactoe

namespace std {

template <>
struct hash<tictactoe::Game::State> {
  size_t operator()(const tictactoe::Game::State& pos) const { return pos.hash(); }
};

}  // namespace std

static_assert(core::concepts::Game<tictactoe::Game>);

#include <inline/games/tictactoe/Game.inl>
