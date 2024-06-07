#pragma once

#include <array>
#include <cstdint>
#include <functional>

#include <boost/functional/hash.hpp>
#include <torch/torch.h>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <core/concepts/Game.hpp>
#include <core/SimpleFullState.hpp>
#include <core/Symmetries.hpp>
#include <core/TrainingTargets.hpp>
#include <mcts/SearchResults.hpp>
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
  static constexpr int kNumPlayers = 2;
  static constexpr int kNumActions = kNumColumns;
  static constexpr int kMaxBranchingFactor = kNumColumns;

  using ActionMask = std::bitset<kNumActions>;
  using player_name_array_t = std::array<std::string, kNumPlayers>;

  using InputShape = Eigen::Sizes<kNumPlayers, kNumRows, kNumColumns>;
  using InputTensor = eigen_util::FTensor<InputShape>;
  using PolicyShape = Eigen::Sizes<kNumColumns>;
  using PolicyTensor = eigen_util::FTensor<PolicyShape>;
  using ValueArray = Eigen::Array<float, kNumPlayers, 1>;
  using ActionOutcome = core::ActionOutcome<ValueArray>;
  using MctsSearchResults = mcts::SearchResults<Game>;

  struct StateSnapshot {
    core::seat_index_t get_current_player() const;
    core::seat_index_t get_player_at(int row, int col) const;
    bool operator==(const StateSnapshot& other) const = default;
    size_t hash() const;

    mask_t full_mask = 0;        // spaces occupied by either player
    mask_t cur_player_mask = 0;  // spaces occupied by current player
  };

  using GameLogReader = core::GameLogReader<Game>;
  using FullState = core::SimpleFullState<StateSnapshot>;

  using Transform = core::Transform<FullState, PolicyTensor>;
  using Identity = core::IdentityTransform<FullState, PolicyTensor>;

  struct Reflect : public core::ReflexiveTransform<FullState, PolicyTensor> {
    void apply(FullState& state) override;
    void apply(PolicyTensor& policy) override;
  };

  using TransformList = mp::TypeList<Identity, Reflect>;
  using SymmetryIndexSet = std::bitset<mp::Length_v<TransformList>>;

  struct Rules {
    static ActionMask legal_moves(const FullState& state);
    static core::seat_index_t current_player(const StateSnapshot&);
    static ActionOutcome apply(FullState& state, core::action_t action);
    static SymmetryIndexSet get_symmetry_indices(const FullState& state);
  };

  struct IO {
    static std::string action_delimiter() { return ""; }
    static std::string action_to_str(core::action_t action) { return std::to_string(action); }
    static void print_snapshot(const StateSnapshot&, core::action_t last_action = -1,
                               const player_name_array_t* player_names = nullptr);
    static void print_mcts_results(const PolicyTensor& action_policy, const MctsSearchResults&);

   private:
    static void print_row(const StateSnapshot&, row_t row, column_t blink_column)
  };

  struct InputTensorizor {
    static InputShape tensorize(const FullState& state);
  };

  struct TrainingTargetTensorizor {
    using BoardShape = Eigen::Sizes<kNumRows, kNumColumns>;

    using PolicyTarget = core::PolicyTarget<PolicyTensor>;
    using ValueTarget = core::ValueTarget<ValueArray>;
    using OppPolicyTarget = core::OppPolicyTarget<PolicyTensor>;

    struct OwnershipTarget {
      static constexpr const char* kName = "ownership";
      using Tensor = eigen_util::FTensor<BoardShape>;
      static Tensor tensorize(const GameLogReader& reader);
    };

    using TargetList = mp::TypeList<PolicyTarget, ValueTarget, OppPolicyTarget, OwnershipTarget>;
  };

 private:
  static constexpr int _to_bit_index(row_t row, column_t col);
  static constexpr mask_t _column_mask(column_t col);
  static constexpr mask_t _bottom_mask(column_t col);
  static constexpr mask_t _full_bottom_mask();
};

static_assert(core::concepts::Game<c4::Game>);

using Player = core::AbstractPlayer<Game>;

}  // namespace c4

#include <inline/games/connect4/Game.inl>
