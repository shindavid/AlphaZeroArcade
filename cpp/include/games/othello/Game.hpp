#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <tuple>

#include <boost/functional/hash.hpp>
#include <torch/torch.h>

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <core/GameLog.hpp>
#include <core/GameTypes.hpp>
#include <core/TrainingTargets.hpp>
#include <games/othello/Constants.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>
#include <util/MetaProgramming.hpp>

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
  struct Constants {
    static constexpr int kNumPlayers = 2;
    static constexpr int kNumActions = othello::kNumGlobalActions;
    static constexpr int kMaxBranchingFactor = othello::kMaxNumLocalActions;
    static constexpr int kHistorySize = 0;
    static constexpr float kOpeningLength = 25.298;  // likely too big, just keeping previous value
  };

  struct BaseState {
    auto operator<=>(const BaseState& other) const = default;
    size_t hash() const;

    mask_t opponent_mask;    // spaces occupied by either player
    mask_t cur_player_mask;  // spaces occupied by current player
    core::seat_index_t cur_player;
    int8_t pass_count;
  };

  using FullState = BaseState;
  using SymmetryGroup = groups::D4;
  using Types = core::GameTypes<Constants, BaseState, SymmetryGroup>;

  struct Symmetries {
    static Types::SymmetryMask get_mask(const BaseState& state);
    static void apply(BaseState& state, group::element_t sym);
    static void apply(Types::PolicyTensor& policy, group::element_t sym);
    static void apply(core::action_t& action, group::element_t sym);
    static group::element_t get_canonical_symmetry(const BaseState& state);
  };

  struct Rules {
    static void init_state(FullState& state, group::element_t sym = group::kIdentity);
    static Types::ActionMask get_legal_moves(const FullState& state);
    static core::seat_index_t get_current_player(const BaseState& state);
    static Types::ActionOutcome apply(FullState& state, core::action_t action);

   private:
    static Types::ValueArray compute_outcome(const FullState& state);
  };

  struct IO {
    static std::string action_delimiter() { return "-"; }
    static std::string action_to_str(core::action_t action);
    static void print_state(std::ostream&, const BaseState&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);
    static void print_mcts_results(std::ostream&, const Types::PolicyTensor& action_policy,
                                   const Types::SearchResults&);

   private:
    static int print_row(char* buf, int n, const BaseState&, const Types::ActionMask&, row_t row,
                         column_t blink_column);
  };

  struct InputTensorizor {
    static constexpr int kDim0 = kNumPlayers * (1 + Constants::kHistorySize);
    using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDimension, kBoardDimension>>;
    using MCTSKey = BaseState;
    using EvalKey = BaseState;

    static MCTSKey mcts_key(const FullState& state) { return state; }
    static EvalKey eval_key(const BaseState* start, const BaseState* cur) { return *cur; }
    static Tensor tensorize(const BaseState* start, const BaseState* cur);
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

 private:
  static int get_count(const BaseState&, core::seat_index_t seat);
  static core::seat_index_t get_player_at(const BaseState&, int row, int col);  // -1 for unoccupied
  static mask_t get_moves(mask_t P, mask_t O);
  static mask_t get_some_moves(mask_t P, mask_t mask, int dir);
};

extern uint64_t (*flip[kNumGlobalActions])(const uint64_t, const uint64_t);

}  // namespace othello

namespace std {

template <>
struct hash<othello::Game::BaseState> {
  size_t operator()(const othello::Game::BaseState& pos) const { return pos.hash(); }
};

}  // namespace std

static_assert(core::concepts::Game<othello::Game>);

#include <inline/games/othello/Game.inl>
