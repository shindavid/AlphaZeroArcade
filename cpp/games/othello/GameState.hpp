#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <tuple>

#include <boost/functional/hash.hpp>
#include <torch/torch.h>

#include <common/SquareBoardSymmetryBase.hpp>
#include <core/AbstractPlayer.hpp>
#include <core/AbstractSymmetryTransform.hpp>
#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/SerializerTypes.hpp>
#include <core/serializers/DeterministicGameSerializer.hpp>
#include <games/othello/Constants.hpp>
#include <mcts/SearchResults.hpp>
#include <mcts/SearchResultsDumper.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

namespace othello { class GameState; }

template <>
struct std::hash<othello::GameState> {
  std::size_t operator()(const othello::GameState& state) const;
};

namespace othello {

/*
 * See <othello/Constants.hpp> for bitboard representation details.
 *
 * The algorithms for manipulating the board are lifted from:
 *
 * https://github.com/abulmo/edax-reversi
 */
class GameState {
public:
  static constexpr int kNumPlayers = othello::kNumPlayers;
  static constexpr int kMaxNumLocalActions = othello::kMaxNumLocalActions;
  static constexpr int kTypicalNumMovesPerGame = othello::kTypicalNumMovesPerGame;
  static constexpr int kMaxNumSymmetries = 8;

  using ActionShape = Eigen::Sizes<othello::kNumGlobalActions>;
  using BoardShape = Eigen::Sizes<kBoardDimension, kBoardDimension>;

  static std::string action_delimiter() { return "-"; }

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
    return common::SquareBoardSymmetries<Tensor, BoardShape>::get_symmetry(index);
  }

  SymmetryIndexSet get_symmetry_indices() const;

  int get_count(core::seat_index_t seat) const;
  core::seat_index_t get_current_player() const { return cur_player_; }
  GameOutcome apply_move(const Action& action);
  ActionMask get_valid_actions() const;
  std::string action_to_str(const Action& action) const;

  core::seat_index_t get_player_at(int row, int col) const;  // -1 for unoccupied
  void dump(const Action* last_action=nullptr, const player_name_array_t* player_names=nullptr) const;
  bool operator==(const GameState& other) const = default;
  std::size_t hash() const;

private:
  auto to_tuple() const { return std::make_tuple(opponent_mask_, cur_player_mask_, cur_player_, pass_count_); }
  GameOutcome compute_outcome() const;  // assumes game has ended
  void row_dump(const ActionMask& valid_actions, row_t row, column_t blink_column) const;
  static mask_t get_moves(mask_t P, mask_t O);
  static mask_t get_some_moves(mask_t P, mask_t mask, int dir);

  mask_t opponent_mask_ = kStartingWhiteMask;  // spaces occupied by either player
  mask_t cur_player_mask_ = kStartingBlackMask;  // spaces occupied by current player
  core::seat_index_t cur_player_ = kStartingColor;
  int8_t pass_count_ = 0;
};

static_assert(core::GameStateConcept<othello::GameState>);

extern uint64_t (*flip[kNumGlobalActions])(const uint64_t, const uint64_t);

using Player = core::AbstractPlayer<GameState>;

}  // namespace c4

namespace core {

// template specialization
template<> struct serializer<othello::GameState> {
  using type = DeterministicGameSerializer<othello::GameState>;
};

}  // namespace core

namespace mcts {

template<> struct SearchResultsDumper<othello::GameState> {
  using LocalPolicyArray = othello::GameState::LocalPolicyArray;
  using SearchResults = mcts::SearchResults<othello::GameState>;

  static void dump(const LocalPolicyArray& action_policy, const SearchResults& results);
};

}

#include <games/othello/inl/GameState.inl>
