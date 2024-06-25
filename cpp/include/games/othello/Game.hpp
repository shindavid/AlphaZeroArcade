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
    static constexpr int kNumSymmetries = 8;
  };

  struct BaseState {
    bool operator==(const BaseState& other) const = default;
    size_t hash() const;

    mask_t opponent_mask = kStartingWhiteMask;    // spaces occupied by either player
    mask_t cur_player_mask = kStartingBlackMask;  // spaces occupied by current player
    core::seat_index_t cur_player = kStartingColor;
    int8_t pass_count = 0;
  };

  using FullState = BaseState;

  using Types = core::GameTypes<Constants, BaseState>;

  using Identity = core::IdentityTransform<BaseState, Types::PolicyTensor>;

  struct Rot90Clockwise : public core::Transform<BaseState, Types::PolicyTensor> {
    void apply(BaseState& pos) override;
    void undo(BaseState& pos) override;
    void apply(Types::PolicyTensor& policy) override;
    void undo(Types::PolicyTensor& policy) override;
  };

  struct Rot180 : public core::ReflexiveTransform<BaseState, Types::PolicyTensor> {
    void apply(BaseState& pos) override;
    void apply(Types::PolicyTensor& policy) override;
  };

  struct Rot270Clockwise : public core::Transform<BaseState, Types::PolicyTensor> {
    void apply(BaseState& pos) override;
    void undo(BaseState& pos) override;
    void apply(Types::PolicyTensor& policy) override;
    void undo(Types::PolicyTensor& policy) override;
  };

  struct FlipVertical : public core::ReflexiveTransform<BaseState, Types::PolicyTensor> {
    void apply(BaseState& pos) override;
    void apply(Types::PolicyTensor& policy) override;
  };

  struct MirrorHorizontal : public core::ReflexiveTransform<BaseState, Types::PolicyTensor> {
    void apply(BaseState& pos) override;
    void apply(Types::PolicyTensor& policy) override;
  };

  struct FlipMainDiag : public core::ReflexiveTransform<BaseState, Types::PolicyTensor> {
    void apply(BaseState& pos) override;
    void apply(Types::PolicyTensor& policy) override;
  };

  struct FlipAntiDiag : public core::ReflexiveTransform<BaseState, Types::PolicyTensor> {
    void apply(BaseState& pos) override;
    void apply(Types::PolicyTensor& policy) override;
  };

  using TransformList = mp::TypeList<Identity, Rot90Clockwise, Rot180, Rot270Clockwise,
                                     FlipVertical, MirrorHorizontal, FlipMainDiag, FlipAntiDiag>;

  struct Rules {
    static Types::ActionMask get_legal_moves(const FullState& state);
    static core::seat_index_t get_current_player(const BaseState& state);
    static Types::ActionOutcome apply(FullState& state, core::action_t action);
    static Types::SymmetryIndexSet get_symmetries(const FullState& state);

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
    using Tensor = eigen_util::FTensor<Eigen::Sizes<kNumPlayers, kBoardDimension, kBoardDimension>>;
    using EvalKey = BaseState;
    using MCTSKey = BaseState;

    static EvalKey eval_key(const FullState& state) { return state; }
    static MCTSKey mcts_key(const FullState& state) { return state; }
    static Tensor tensorize(const BaseState* start, const BaseState* cur);
  };

  struct TrainingTargets {
    using BoardShape = Eigen::Sizes<kBoardDimension, kBoardDimension>;

    using PolicyTarget = core::PolicyTarget<Game>;
    using ValueTarget = core::ValueTarget<Game>;
    using OppPolicyTarget = core::OppPolicyTarget<Game>;

    struct ScoreMarginTarget {
      static constexpr const char* kName = "score_margin";
      using Tensor = eigen_util::FTensor<Eigen::Sizes<1>>;

      static Tensor tensorize(const Types::GameLogView& view);
    };

    struct OwnershipTarget {
      static constexpr const char* kName = "ownership";
      using Tensor = eigen_util::FTensor<BoardShape>;

      static Tensor tensorize(const Types::GameLogView& view);
    };

    using List = mp::TypeList<PolicyTarget, ValueTarget, OppPolicyTarget, ScoreMarginTarget,
                              OwnershipTarget>;
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
