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
#include <games/othello/Constants.hpp>
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
#include <tuple>

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
    using kNumActionsPerMode = util::int_sequence<othello::kNumGlobalActions>;
    static constexpr int kNumPlayers = 2;
    static constexpr int kMaxBranchingFactor = othello::kMaxNumLocalActions;
  };

  struct MctsConfiguration : public core::MctsConfigurationBase {
    static constexpr float kOpeningLength = 25.298;  // likely too big, just keeping previous value
  };

  struct State {
    auto operator<=>(const State& other) const = default;
    size_t hash() const;

    mask_t opponent_mask;    // spaces occupied by either player
    mask_t cur_player_mask;  // spaces occupied by current player
    core::seat_index_t cur_player;
    int8_t pass_count;
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

  struct Rules : public game_base::RulesBase<Types, State> {
    static void init_state(State&);
    static Types::ActionMask get_legal_moves(const StateHistory&);
    static core::action_mode_t get_action_mode(const State&) { return 0; }
    static core::seat_index_t get_current_player(const State&);
    static void apply(StateHistory&, core::action_t action);
    static bool is_terminal(const State& state, core::seat_index_t last_player,
                            core::action_t last_action, GameResults::Tensor& outcome);
    static Types::ActionMask get_legal_moves(const State&);

   private:
    static GameResults::Tensor compute_outcome(const State& state);
  };

  struct IO : public core::IOBase<Types, State> {
    static std::string action_delimiter() { return "-"; }
    static std::string action_to_str(core::action_t action, core::action_mode_t);
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);
    static void print_mcts_results(std::ostream&, const Types::PolicyTensor& action_policy,
                                   const Types::SearchResults&);

   private:
    static int print_row(char* buf, int n, const State&, const Types::ActionMask&, row_t row,
                         column_t blink_column);
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
    using ScoreMarginShape = Eigen::Sizes<2, 2 * kNumCells + 1>;  // pdf/cdf, score-margin

    using PolicyTarget = core::PolicyTarget<Game>;
    using ValueTarget = core::ValueTarget<Game>;
    using ActionValueTarget = core::ActionValueTarget<Game>;
    using OppPolicyTarget = core::OppPolicyTarget<Game>;

    struct ScoreMarginTarget {
      static constexpr const char* kName = "score_margin";
      using Tensor = eigen_util::FTensor<ScoreMarginShape>;

      static Tensor tensorize(const Types::GameLogView& view);
    };

    struct OwnershipTarget {
      static constexpr const char* kName = "ownership";
      using Tensor = eigen_util::FTensor<OwnershipShape>;

      static Tensor tensorize(const Types::GameLogView& view);
    };

    using List = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget, OppPolicyTarget,
                              ScoreMarginTarget, OwnershipTarget>;
  };

  static void static_init() {}

 private:
  static int get_count(const State&, core::seat_index_t seat);
  static core::seat_index_t get_player_at(const State&, int row, int col);  // -1 for unoccupied
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

#include <inline/games/othello/Game.inl>
