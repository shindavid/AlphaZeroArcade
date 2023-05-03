#pragma once

#include <array>
#include <cstdint>
#include <functional>

#include <boost/functional/hash.hpp>
#include <torch/torch.h>

#include <common/AbstractPlayer.hpp>
#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/MctsResults.hpp>
#include <common/SerializerTypes.hpp>
#include <common/serializers/DeterministicGameSerializer.hpp>
#include <connect4/Constants.hpp>
#include <util/EigenUtil.hpp>

namespace c4 { class GameState; }

template <>
struct std::hash<c4::GameState> {
  std::size_t operator()(const c4::GameState& state) const;
};

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
 * Unlike the connect4 package, we use 0-indexing for column indices.
 */
class GameState {
public:
  static constexpr int kNumPlayers = c4::kNumPlayers;
  static constexpr int kNumGlobalActions = kNumColumns;
  static constexpr int kMaxNumLocalActions = kNumColumns;
  static constexpr int kTypicalNumMovesPerGame = 40;

  using GameStateTypes = common::GameStateTypes<GameState>;
  using ActionMask = GameStateTypes::ActionMask;
  using player_name_array_t = GameStateTypes::player_name_array_t;
  using ValueProbDistr = GameStateTypes::ValueProbDistr;
  using MctsResults = common::MctsResults<GameState>;
  using LocalPolicyProbDistr = GameStateTypes::LocalPolicyProbDistr;
  using GameOutcome = GameStateTypes::GameOutcome;

  common::seat_index_t get_current_player() const;
  GameOutcome apply_move(common::action_index_t action);
  ActionMask get_valid_actions() const;
  int get_move_number() const;

  template<eigen_util::FixedTensorConcept InputSlab> void tensorize(InputSlab&) const;
  void dump(common::action_index_t last_action=-1, const player_name_array_t* player_names=nullptr) const;
  bool operator==(const GameState& other) const;
  std::size_t hash() const { return boost::hash_range(&full_mask_, (&full_mask_) + 2); }

  static void dump_mcts_output(const ValueProbDistr& mcts_value, const LocalPolicyProbDistr& mcts_policy,
                                const MctsResults& results);

private:
  void row_dump(row_t row, column_t blink_column) const;

  static constexpr int _to_bit_index(column_t col, row_t row);
  static constexpr mask_t _column_mask(column_t col);  // mask containing piece on all cells of given column
  static constexpr mask_t _bottom_mask(column_t col);  // mask containing single piece at bottom cell
  static constexpr mask_t _full_bottom_mask();  // mask containing piece in each bottom cell

  mask_t full_mask_ = 0;  // spaces occupied by either player
  mask_t cur_player_mask_ = 0;  // spaces occupied by current player
};

static_assert(common::GameStateConcept<c4::GameState>);

using Player = common::AbstractPlayer<GameState>;

}  // namespace c4

namespace common {

// template specialization
template<> struct serializer<c4::GameState> {
  using type = DeterministicGameSerializer<c4::GameState>;
};

}  // namespace common

#include <connect4/inl/GameState.inl>
