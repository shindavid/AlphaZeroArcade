#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <core/GameLog.hpp>
#include <core/GameTypes.hpp>
#include <core/IOBase.hpp>
#include <core/SimpleStateHistory.hpp>
#include <core/TrainingTargets.hpp>
#include <core/TrivialSymmetries.hpp>
#include <core/WinShareResults.hpp>
#include <games/blokus/Constants.hpp>
#include <games/blokus/GameState.hpp>
#include <games/blokus/Types.hpp>
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

namespace blokus {

class Game {
 public:
  struct Constants {
    static constexpr int kNumPlayers = blokus::kNumPlayers;
    static constexpr int kNumActions = blokus::kNumActions;
    static constexpr int kMaxBranchingFactor = blokus::kNumPieceOrientationCorners;
    static constexpr int kNumPreviousStatesToEncode = 0;
    static constexpr float kOpeningLength = 70.314;  // likely too big, just keeping previous value
  };

  using State = blokus::GameState;
  using GameResults = core::WinShareResults<Constants::kNumPlayers>;
  using StateHistory = core::SimpleStateHistory<State, Constants::kNumPreviousStatesToEncode>;

  /*
   * After the initial placement of the first piece, the rules of the game are symmetric. But the
   * rules are not symmetric for the first piece placement, and as a result, strategic
   * considerations are asymmetric for much, if not all of the game. Because of this, it's unclear
   * whether exploiting symmetry will be useful, so we use the trivial group.
   */
  using SymmetryGroup = groups::TrivialGroup;
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;
  using Symmetries = core::TrivialSymmetries;
  using MCTSKey = State;

  struct Rules {
    static void init_state(State&);
    static Types::ActionMask get_legal_moves(const StateHistory&);
    static core::seat_index_t get_current_player(const State&);
    static void apply(StateHistory&, core::action_t action);
    static bool is_terminal(const State& state, core::seat_index_t last_player,
                            core::action_t last_action, GameResults::Tensor& outcome);

   private:
    static GameResults::Tensor compute_outcome(const State& state);
  };

  struct IO : core::IOBase<Types, State, MCTSKey> {
    static std::string action_delimiter() { return "-"; }
    static std::string action_to_str(core::action_t action);
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);
    static void print_mcts_results(std::ostream&, const Types::PolicyTensor& action_policy,
                                   const Types::SearchResults&);

    /*
     * Inverse operation of print_state(ss, state) in non-tty-mode.
     *
     * Assumes that the last pass_count players have passed.
     */
    static State load(const std::string& str, int pass_count=0);
  };

  // TODO: add unplayed-pieces as an auxiliary input.
  struct InputTensorizor {
    // +1 to record the partial move if necessary.
    static constexpr int kDim0 = kNumPlayers * (1 + Constants::kNumPreviousStatesToEncode) + 1;
    using Tensor =
        eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDimension, kBoardDimension>>;
    using MCTSKey = State;
    using EvalKey = State;

    static MCTSKey mcts_key(const StateHistory& history) { return history.current(); }
    template <typename Iter> static EvalKey eval_key(Iter start, Iter cur) { return *cur; }
    template <typename Iter> static Tensor tensorize(Iter start, Iter cur);
  };

  struct TrainingTargets {
    using BoardShape = Eigen::Sizes<kBoardDimension, kBoardDimension>;
    using OwnershipShape = Eigen::Sizes<kNumPlayers + 1, kBoardDimension, kBoardDimension>;
    using ScoreShape = Eigen::Sizes<2, kVeryBadScore + 1, kNumPlayers>;  // pdf/cdf, score, player
    using UnplayedPiecesShape = Eigen::Sizes<kNumPlayers, kNumPieces>;

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

    /*
     * Which pieces are unplayed at the end of the game.
     */
    struct UnplayedPiecesTarget {
      static constexpr const char* kName = "unplayed_pieces";
      using Tensor = eigen_util::FTensor<UnplayedPiecesShape>;

      static Tensor tensorize(const Types::GameLogView& view);
    };

    // TODO:
    // - ReachableSquaresTarget: for each square, whether it is reachable by some player if all
    //                           other players are forced to pass all their turns.
    // - OpponentReplySquaresTarget: for each square, whether some opponent plays a piece there
    //                               before the current player's next move.

    using List = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget, ScoreTarget,
                              OwnershipTarget, UnplayedPiecesTarget>;
  };

  static void static_init() {}
};

}  // namespace blokus

namespace std {

template <>
struct hash<blokus::Game::State> {
  size_t operator()(const blokus::Game::State& pos) const { return pos.hash(); }
};

}  // namespace std

static_assert(core::concepts::Game<blokus::Game>);

#include <inline/games/blokus/Game.inl>
