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
#include <games/connect4/Constants.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>
#include <util/MetaProgramming.hpp>

namespace c4 {

/*
 * Bit order encoding for the board:
 *
 *  5 13 21 29 37 45 53
 *  4 12 20 28 36 44 52
 *  3 11 19 27 35 43 51
 *  2 10 18 26 34 42 50
 *  1  9 17 25 33 41 49
 *  0  8 16 24 32 40 48
 *
 * Based on https://github.com/PascalPons/connect4
 *
 * Unlike the PascalPons package, we use 0-indexing for column indices.
 */
struct Game {
  struct Constants {
    static constexpr int kNumPlayers = 2;
    static constexpr int kNumActions = kNumColumns;
    static constexpr int kMaxBranchingFactor = kNumColumns;
    static constexpr int kHistorySize = 0;
  };

  struct BaseState {
    bool operator==(const BaseState& other) const = default;
    size_t hash() const;

    mask_t full_mask;        // spaces occupied by either player
    mask_t cur_player_mask;  // spaces occupied by current player
  };

  using FullState = BaseState;
  using SymmetryGroup = groups::D1;
  using Types = core::GameTypes<Constants, BaseState, SymmetryGroup>;

  struct Symmetries {
    static Types::SymmetryMask get_mask(const BaseState& state);
    static void apply(BaseState& state, group::element_t sym);
    static void apply(Types::PolicyTensor& policy, group::element_t sym);
    static void apply(core::action_t& action, group::element_t sym);
  };

  struct Rules {
    static void init_state(FullState& state);
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

   private:
    static int print_row(char* buf, int n, const BaseState&, row_t row, column_t blink_column);
  };

  struct InputTensorizor {
    using Tensor = eigen_util::FTensor<Eigen::Sizes<kNumPlayers, kNumRows, kNumColumns>>;
    using MCTSKey = BaseState;
    using EvalKey = BaseState;

    static MCTSKey mcts_key(const FullState& state) { return state; }
    static EvalKey eval_key(const BaseState* start, const BaseState* cur) { return *cur; }
    static Tensor tensorize(const BaseState* start, const BaseState* cur);
  };

  struct TrainingTargets {
    using BoardShape = Eigen::Sizes<kNumRows, kNumColumns>;

    using PolicyTarget = core::PolicyTarget<Game>;
    using ValueTarget = core::ValueTarget<Game>;
    using OppPolicyTarget = core::OppPolicyTarget<Game>;

    struct OwnershipTarget {
      static constexpr const char* kName = "ownership";
      using Tensor = eigen_util::FTensor<BoardShape>;

      static Tensor tensorize(const Types::GameLogView& view);
    };

    using List = mp::TypeList<PolicyTarget, ValueTarget, OppPolicyTarget, OwnershipTarget>;
  };

 private:
  static core::seat_index_t _get_player_at(const BaseState& state, row_t row, column_t col);
  static constexpr int _to_bit_index(row_t row, column_t col);
  static constexpr mask_t _column_mask(column_t col);
  static constexpr mask_t _bottom_mask(column_t col);
  static constexpr mask_t _full_bottom_mask();
};

}  // namespace c4

namespace std {

template <>
struct hash<c4::Game::BaseState> {
  size_t operator()(const c4::Game::BaseState& pos) const { return pos.hash(); }
};

}  // namespace std

static_assert(core::concepts::Game<c4::Game>);

#include <inline/games/connect4/Game.inl>
