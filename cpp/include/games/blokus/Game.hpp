#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>

#include <boost/functional/hash.hpp>
#include <torch/torch.h>

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <core/GameLog.hpp>
#include <core/GameTypes.hpp>
#include <core/TrainingTargets.hpp>
#include <games/blokus/Constants.hpp>
#include <games/blokus/Types.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>
#include <util/MetaProgramming.hpp>

namespace blokus {

class Game {
 public:
  struct Constants {
    static constexpr int kNumPlayers = blokus::kNumPlayers;
    static constexpr int kNumActions = blokus::kNumActions;
    static constexpr int kMaxBranchingFactor = blokus::kNumPieceOrientationCorners;
    static constexpr int kHistorySize = 0;
  };

  /*
   * BaseState is split internally into two parts: core_t and aux_t.
   *
   * core_t unamgibuously represents the game state.
   *
   * aux_t contains additional information that can be computed from core_t, but is stored for
   * efficiency.
   *
   * TODO: use Zobrist-hashing to speed-up hashing.
   */
  struct BaseState {
    auto operator<=>(const BaseState& other) const { return core <=> other.core; }
    bool operator==(const BaseState& other) const { return core == other.core; }
    bool operator!=(const BaseState& other) const { return core != other.core; }
    size_t hash() const;
    int remaining_square_count(color_t) const;
    color_t last_placed_piece_color() const;
    int pass_count() const { return core.pass_count; }

    /*
     * Sets this->aux from this->core.
     */
    void compute_aux();

    /*
     * Throws an exception if aux is not consistent with core.
     */
    void validate_aux() const;

    // core_t unambiguously represents the game state.
    struct core_t {
      auto operator<=>(const core_t& other) const = default;
      BitBoard occupied_locations[kNumColors];

      color_t cur_color;
      int8_t pass_count;

      // We split the move into multiple parts:
      // 1. The location of a piece corner
      // 2. The piece/orientation/square to place on that location
      //
      // This value stores part 1. A location of (-1, -1) indicates that the player has not
      // selected a location.
      Location partial_move;
    };

    // TODO: consider moving some of these members into FullState. The ones that support
    // tensorization should stay here, but the ones that only facilitate rules-calculations can
    // be moved to FullState to reduce the disk footprint of game logs.
    struct aux_t {
      auto operator<=>(const aux_t&) const = default;
      PieceMask played_pieces[kNumColors];
      BitBoard unplayable_locations[kNumColors];
      BitBoard corner_locations[kNumColors];
    };

    core_t core;
    aux_t aux;
  };

  using FullState = BaseState;

  /*
   * After the initial placement of the first piece, the rules of the game are symmetric. But the
   * rules are not symmetric for the first piece placement, and as a result, strategic
   * considerations are asymmetric for much, if not all of the game. Because of this, it's unclear
   * whether exploiting symmetry will be useful, so we use the trivial group.
   */
  using SymmetryGroup = groups::TrivialGroup;
  using Types = core::GameTypes<Constants, BaseState, SymmetryGroup>;

  struct Symmetries {
    static Types::SymmetryMask get_mask(const BaseState& state);
    static void apply(BaseState& state, group::element_t sym) {}
    static void apply(Types::PolicyTensor& policy, group::element_t sym) {}
    static void apply(core::action_t& action, group::element_t sym) {}
    static group::element_t get_canonical_symmetry(const BaseState& state) { return 0; }
  };

  struct Rules {
    static void init_state(FullState& state, group::element_t sym = group::kIdentity);
    static Types::ActionMask get_legal_moves(const FullState& state);
    static core::seat_index_t get_current_player(const BaseState&);
    static Types::ActionOutcome apply(FullState& state, core::action_t action);

   private:
    static Types::ActionOutcome compute_outcome(const FullState& state);
  };

  struct IO {
    static std::string action_delimiter() { return "-"; }
    static std::string action_to_str(core::action_t action);
    static void print_state(std::ostream&, const BaseState&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);
    static void print_mcts_results(std::ostream&, const Types::PolicyTensor& action_policy,
                                   const Types::SearchResults&);

    /*
     * Inverse operation of print_state(ss, state) in non-tty-mode.
     *
     * Assumes that the last pass_count players have passed.
     */
    static FullState load(const std::string& str, int pass_count=0);
  };

  struct InputTensorizor {
    // +1 to record the partial move if necessary.
    using Tensor =
        eigen_util::FTensor<Eigen::Sizes<kNumPlayers + 1, kBoardDimension, kBoardDimension>>;
    using MCTSKey = BaseState;
    using EvalKey = BaseState;

    static MCTSKey mcts_key(const FullState& state) { return state; }
    static EvalKey eval_key(const BaseState* start, const BaseState* cur) { return *cur; }
    static Tensor tensorize(const BaseState* start, const BaseState* cur);
  };

  struct TrainingTargets {
    using BoardShape = Eigen::Sizes<kBoardDimension, kBoardDimension>;
    using OwnershipShape = Eigen::Sizes<kNumPlayers + 1, kBoardDimension, kBoardDimension>;
    using ScoreShape = Eigen::Sizes<2, kMaxScore + 1, kNumPlayers>;  // pdf/cdf, score, player

    using PolicyTarget = core::PolicyTarget<Game>;
    using ValueTarget = core::ValueTarget<Game>;
    using ActionValueTarget = core::ActionValueTarget<Game>;

    struct ScoreTarget {
      static constexpr const char* kName = "score";
      using Tensor = eigen_util::FTensor<ScoreShape>;

      static Tensor tensorize(const Types::GameLogView& view);
    };

    /*
     * Who owns which square at the end of the game.
     */
    struct OwnershipTarget {
      static constexpr const char* kName = "ownership";
      using Tensor = eigen_util::FTensor<OwnershipShape>;

      static Tensor tensorize(const Types::GameLogView& view);
    };

    struct DummyScoreTarget {
      static constexpr const char* kName = "dummy-score";
      using Tensor = eigen_util::FTensor<ScoreShape>;

      static Tensor tensorize(const Types::GameLogView& view);
    };

    struct DummyOwnershipTarget {
      static constexpr const char* kName = "dummy-ownership";
      using Tensor = eigen_util::FTensor<OwnershipShape>;

      static Tensor tensorize(const Types::GameLogView& view);
    };

    // TODO:
    // - UnplayedPiecesTarget
    // - ReachableSquaresTarget: for each square, whether it is reachable by some player if all
    //                           other players are forced to pass all their turns.
    // - OpponentReplySquaresTarget: for each square, whether some opponent plays a piece there
    //                               before the current player's next move.

    using List = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget, ScoreTarget,
                              OwnershipTarget, DummyScoreTarget, DummyOwnershipTarget>;
  };
};

}  // namespace blokus

namespace std {

template <>
struct hash<blokus::Game::BaseState> {
  size_t operator()(const blokus::Game::BaseState& pos) const { return pos.hash(); }
};

}  // namespace std

static_assert(core::concepts::Game<blokus::Game>);

#include <inline/games/blokus/Game.inl>
