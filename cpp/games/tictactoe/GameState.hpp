#pragma once

#include <array>
#include <cstdint>
#include <functional>

#include <boost/functional/hash.hpp>
#include <torch/torch.h>

#include <core/SquareBoardSymmetries.hpp>
#include <core/AbstractPlayer.hpp>
#include <core/AbstractSymmetryTransform.hpp>
#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/SerializerTypes.hpp>
#include <core/serializers/DeterministicGameSerializer.hpp>
#include <games/tictactoe/Constants.hpp>
#include <mcts/SearchResults.hpp>
#include <mcts/SearchResultsDumper.hpp>
#include <util/EigenUtil.hpp>

namespace tictactoe {
class GameState;
}

template <>
struct std::hash<tictactoe::GameState> {
  std::size_t operator()(const tictactoe::GameState& state) const;
};

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
class GameState {
 public:
  static constexpr int kNumPlayers = tictactoe::kNumPlayers;
  static constexpr int kMaxNumLocalActions = kNumCells;
  static constexpr int kTypicalNumMovesPerGame = kNumCells;
  static constexpr int kMaxNumSymmetries = 8;

  using ActionShape = Eigen::Sizes<kNumCells>;
  using BoardShape = Eigen::Sizes<kBoardDimension, kBoardDimension>;

  static std::string action_delimiter() { return ""; }

  using GameStateTypes = core::GameStateTypes<GameState>;
  using SymmetryIndexSet = GameStateTypes::SymmetryIndexSet;
  using Action = GameStateTypes::Action;
  using ActionMask = GameStateTypes::ActionMask;
  using player_name_array_t = GameStateTypes::player_name_array_t;
  using ValueArray = GameStateTypes::ValueArray;
  using LocalPolicyArray = GameStateTypes::LocalPolicyArray;
  using GameOutcome = GameStateTypes::GameOutcome;

  template <eigen_util::FixedTensorConcept Tensor>
  core::AbstractSymmetryTransform<Tensor>* get_symmetry(core::symmetry_index_t index) const {
    return core::SquareBoardSymmetries<Tensor, BoardShape>::get_symmetry(index);
  }

  SymmetryIndexSet get_symmetry_indices() const;

  core::seat_index_t get_current_player() const;
  GameOutcome apply_move(const Action& action);
  ActionMask get_valid_actions() const;
  int get_move_number() const;
  mask_t get_current_player_mask() const { return cur_player_mask_; }
  mask_t get_opponent_mask() const { return full_mask_ ^ cur_player_mask_; }
  std::string action_to_str(const Action& action) const { return std::to_string(action[0]); }

  core::seat_index_t get_player_at(int row, int col) const;
  void dump(const Action* last_action = nullptr,
            const player_name_array_t* player_names = nullptr) const;
  bool operator==(const GameState& other) const = default;
  std::size_t hash() const { return boost::hash_range(&full_mask_, (&full_mask_) + 2); }

  static constexpr mask_t kThreeInARowMasks[] = {
      make_mask(0, 1, 2), make_mask(3, 4, 5), make_mask(6, 7, 8), make_mask(0, 3, 6),
      make_mask(1, 4, 7), make_mask(2, 5, 8), make_mask(0, 4, 8), make_mask(2, 4, 6)};

 private:
  mask_t full_mask_ = 0;        // spaces occupied by either player
  mask_t cur_player_mask_ = 0;  // spaces occupied by current player
};

static_assert(core::GameStateConcept<tictactoe::GameState>);

using Player = core::AbstractPlayer<GameState>;

}  // namespace tictactoe

namespace core {

// template specialization
template <>
struct serializer<tictactoe::GameState> {
  using type = DeterministicGameSerializer<tictactoe::GameState>;
};

}  // namespace core

namespace mcts {

template <>
struct SearchResultsDumper<tictactoe::GameState> {
  using LocalPolicyArray = tictactoe::GameState::LocalPolicyArray;
  using SearchResults = mcts::SearchResults<tictactoe::GameState>;

  static void dump(const LocalPolicyArray& action_policy, const SearchResults& results);
};

}  // namespace mcts

#include <games/tictactoe/inl/GameState.inl>
