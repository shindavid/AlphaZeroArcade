#pragma once

#include <array>
#include <cstdint>
#include <functional>

#include <boost/functional/hash.hpp>
#include <torch/torch.h>

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <core/GameLog.hpp>
#include <core/GameTypes.hpp>
#include <core/Symmetries.hpp>
#include <core/TrainingTargets.hpp>
#include <core/SearchResults.hpp>
#include <util/EigenUtil.hpp>

#include <games/connect4/Constants.hpp>

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

    mask_t full_mask = 0;        // spaces occupied by either player
    mask_t cur_player_mask = 0;  // spaces occupied by current player
  };

  using FullState = BaseState;

  using Types = core::GameTypes<Game>;

  using Transform = core::Transform<BaseState, Types::PolicyTensor>;
  using Identity = core::IdentityTransform<BaseState, Types::PolicyTensor>;

  struct Reflect : public core::ReflexiveTransform<BaseState, Types::PolicyTensor> {
    void apply(BaseState& pos) override;
    void apply(Types::PolicyTensor& policy) override;
  };

  using TransformList = mp::TypeList<Identity, Reflect>;
  using SymmetryIndexSet = std::bitset<mp::Length_v<TransformList>>;

  struct Rules {
    static Types::ActionMask get_legal_moves(const FullState& state);
    static core::seat_index_t get_current_player(const BaseState&);
    static Types::ActionOutcome apply(FullState& state, core::action_t action);
    static SymmetryIndexSet get_symmetry_indices(const FullState& state);
  };

  struct IO {
    static std::string action_delimiter() { return ""; }
    static std::string action_to_str(core::action_t action) { return std::to_string(action); }
    static void print_state(const BaseState&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);
    static void print_mcts_results(const Types::PolicyTensor& action_policy,
                                   const Types::SearchResults&);

   private:
    static void print_row(const BaseState&, row_t row, column_t blink_column);
  };

  struct InputTensorizor {
    using Tensor = eigen_util::FTensor<Eigen::Sizes<kNumPlayers, kNumRows, kNumColumns>>;
    using EvalKey = BaseState;
    using MCTSKey = BaseState;

    static EvalKey eval_key(const FullState& state) { return state; }
    static MCTSKey mcts_key(const FullState& state) { return state; }
    static Tensor tensorize(const BaseState* start, const BaseState* cur);
  };

  struct TrainingTargetTensorizor {
    using BoardShape = Eigen::Sizes<kNumRows, kNumColumns>;

    using PolicyTarget = core::PolicyTarget<Game>;
    using ValueTarget = core::ValueTarget<Game>;
    using OppPolicyTarget = core::OppPolicyTarget<Game>;

    struct OwnershipTarget {
      static constexpr const char* kName = "ownership";
      using Tensor = eigen_util::FTensor<BoardShape>;

      static Tensor tensorize(const Types::GameLogView& view);
    };

    using TargetList = mp::TypeList<PolicyTarget, ValueTarget, OppPolicyTarget, OwnershipTarget>;
  };

 private:
  static core::seat_index_t _get_player_at(const BaseState& state, row_t row, column_t col);
  static constexpr int _to_bit_index(row_t row, column_t col);
  static constexpr mask_t _column_mask(column_t col);
  static constexpr mask_t _bottom_mask(column_t col);
  static constexpr mask_t _full_bottom_mask();
};

static_assert(core::concepts::Game<c4::Game>);

}  // namespace c4

namespace std {

template <>
struct hash<c4::Game::BaseState> {
  size_t operator()(const c4::Game::BaseState& pos) const { return pos.hash(); }
};

}  // namespace std

#include <inline/games/connect4/Game.inl>
