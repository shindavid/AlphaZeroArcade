#pragma once

#include <array>
#include <cstdint>
#include <functional>

#include <boost/functional/hash.hpp>
#include <torch/torch.h>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/SerializerTypes.hpp>
#include <core/serializers/DeterministicGameSerializer.hpp>
#include <games/tictactoe/Constants.hpp>
#include <mcts/SearchResults.hpp>
#include <mcts/SearchResultsDumper.hpp>
#include <util/EigenUtil.hpp>

namespace tictactoe { class GameState; }

template <>
struct std::hash<tictactoe::GameState> {
  std::size_t operator()(const tictactoe::GameState& state) const;
};

namespace tictactoe {

/*
 * Bit order encoding for the board:
 *
 * 6 7 8
 * 3 4 5
 * 0 1 2
 */
class GameState {
public:
  using PolicyShape = eigen_util::Shape<kBoardDimension, kBoardDimension>;
  static constexpr int kNumPlayers = tictactoe::kNumPlayers;
  static constexpr int kMaxNumLocalActions = kNumCells;
  static constexpr int kTypicalNumMovesPerGame = kNumCells;

  using GameStateTypes = core::GameStateTypes<GameState>;
  using ActionMask = GameStateTypes::ActionMask;
  using player_name_array_t = GameStateTypes::player_name_array_t;
  using ValueArray = GameStateTypes::ValueArray;
  using LocalPolicyArray = GameStateTypes::LocalPolicyArray;
  using GameOutcome = GameStateTypes::GameOutcome;

  core::seat_index_t get_current_player() const;
  GameOutcome apply_move(core::action_t action);
  ActionMask get_valid_actions() const;
  int get_move_number() const;

  template<eigen_util::FixedTensorConcept InputTensor> void tensorize(InputTensor&) const;
  void dump(core::action_t last_action=-1, const player_name_array_t* player_names=nullptr) const;
  bool operator==(const GameState& other) const = default;
  std::size_t hash() const { return boost::hash_range(&full_mask_, (&full_mask_) + 2); }

private:
  static constexpr mask_t make_mask(int a, int b, int c) {
    return (mask_t(1) << a) + (mask_t(1) << b) + (mask_t(1) << c);
  }

  mask_t full_mask_ = 0;  // spaces occupied by either player
  mask_t cur_player_mask_ = 0;  // spaces occupied by current player
};

static_assert(core::GameStateConcept<tictactoe::GameState>);

using Player = core::AbstractPlayer<GameState>;

}  // namespace tictactoe

namespace core {

// template specialization
template<> struct serializer<tictactoe::GameState> {
  using type = DeterministicGameSerializer<tictactoe::GameState>;
};

}  // namespace core

namespace mcts {

template<> struct SearchResultsDumper<tictactoe::GameState> {
  using LocalPolicyArray = tictactoe::GameState::LocalPolicyArray;
  using SearchResults = mcts::SearchResults<tictactoe::GameState>;

  static void dump(const LocalPolicyArray& action_policy, const SearchResults& results);
};

}  // namespace mcts

#include <games/tictactoe/inl/GameState.inl>
