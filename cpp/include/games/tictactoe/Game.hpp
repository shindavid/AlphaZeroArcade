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
#include <games/tictactoe/Constants.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>
#include <util/MetaProgramming.hpp>

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
  struct Constants {
    static constexpr int kNumPlayers = tictactoe::kNumPlayers;
    static constexpr int kNumActions = tictactoe::kNumCells;
    static constexpr int kMaxBranchingFactor = tictactoe::kNumCells;
    static constexpr int kHistorySize = 0;
  };

  struct BaseState {
    auto operator<=>(const BaseState& other) const = default;
    size_t hash() const;
    mask_t opponent_mask() const { return full_mask ^ cur_player_mask; }

    mask_t full_mask;        // spaces occupied by either player
    mask_t cur_player_mask;  // spaces occupied by current player
  };

  struct FullState : public BaseState {};  // trivial-inheritance to test mcts-second-pass-logic
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
    static core::seat_index_t get_current_player(const BaseState&);
    static Types::ActionOutcome apply(FullState& state, core::action_t action);
  };

  struct IO {
    static std::string action_delimiter() { return ""; }
    static std::string action_to_str(core::action_t action) { return std::to_string(action); }
    static void print_state(std::ostream&, const BaseState&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);
    static void print_mcts_results(std::ostream&, const Types::PolicyTensor& action_policy,
                                   const Types::SearchResults&);
  };

  struct InputTensorizor {
    using Tensor = eigen_util::FTensor<Eigen::Sizes<kNumPlayers, kBoardDimension, kBoardDimension>>;
    using MCTSKey = BaseState;
    using EvalKey = BaseState;

    static MCTSKey mcts_key(const FullState& state) { return state; }
    static EvalKey eval_key(const BaseState* start, const BaseState* cur) { return *cur; }
    static Tensor tensorize(const BaseState* start, const BaseState* cur);
  };

  struct TrainingTargets {
    using BoardShape = Eigen::Sizes<kBoardDimension, kBoardDimension>;

    using PolicyTarget = core::PolicyTarget<Game>;
    using ValueTarget = core::ValueTarget<Game>;
    using OppPolicyTarget = core::OppPolicyTarget<Game>;

    // TODO(FIXME): change this to be 1-hot
    struct OwnershipTarget {
      static constexpr const char* kName = "ownership";
      using Tensor = eigen_util::FTensor<BoardShape>;

      static Tensor tensorize(const Types::GameLogView& view);
    };

    using List = mp::TypeList<PolicyTarget, ValueTarget, OppPolicyTarget, OwnershipTarget>;
  };

  static constexpr mask_t kThreeInARowMasks[] = {
      make_mask(0, 1, 2), make_mask(3, 4, 5), make_mask(6, 7, 8), make_mask(0, 3, 6),
      make_mask(1, 4, 7), make_mask(2, 5, 8), make_mask(0, 4, 8), make_mask(2, 4, 6)};

 private:
  static core::seat_index_t _get_player_at(const BaseState& state, int row, int col);
};

}  // namespace tictactoe

namespace std {

template <>
struct hash<tictactoe::Game::BaseState> {
  size_t operator()(const tictactoe::Game::BaseState& pos) const { return pos.hash(); }
};

}  // namespace std

static_assert(core::concepts::Game<tictactoe::Game>);

#include <inline/games/tictactoe/Game.inl>
