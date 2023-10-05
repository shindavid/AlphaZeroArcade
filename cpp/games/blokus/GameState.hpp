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
#include <games/blokus/Constants.hpp>
#include <mcts/SearchResults.hpp>
#include <mcts/SearchResultsDumper.hpp>
#include <util/EigenUtil.hpp>

namespace blokus { class GameState; }

template <>
struct std::hash<blokus::GameState> {
  std::size_t operator()(const blokus::GameState& state) const;
};

namespace blokus {

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
  static constexpr int kNumPlayers = blokus::kNumPlayers;
  static constexpr int kNumGlobalActions = kNumCells;
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
  mask_t get_current_player_mask() const { return cur_player_mask_; }
  mask_t get_opponent_mask() const { return full_mask_ ^ cur_player_mask_; }

  core::seat_index_t get_player_at(int row, int col) const;
  void dump(core::action_t last_action=-1, const player_name_array_t* player_names=nullptr) const;
  bool operator==(const GameState& other) const = default;
  std::size_t hash() const { return boost::hash_range(&full_mask_, (&full_mask_) + 2); }

  static constexpr mask_t kThreeInARowMasks[] = {
    make_mask(0, 1, 2),
    make_mask(3, 4, 5),
    make_mask(6, 7, 8),
    make_mask(0, 3, 6),
    make_mask(1, 4, 7),
    make_mask(2, 5, 8),
    make_mask(0, 4, 8),
    make_mask(2, 4, 6)
  };

private:
  mask_t full_mask_ = 0;  // spaces occupied by either player
  mask_t cur_player_mask_ = 0;  // spaces occupied by current player
};

static_assert(core::GameStateConcept<blokus::GameState>);

using Player = core::AbstractPlayer<GameState>;

}  // namespace blokus

namespace core {

// template specialization
template<> struct serializer<blokus::GameState> {
  using type = DeterministicGameSerializer<blokus::GameState>;
};

}  // namespace core

namespace mcts {

template<> struct SearchResultsDumper<blokus::GameState> {
  using LocalPolicyArray = blokus::GameState::LocalPolicyArray;
  using SearchResults = mcts::SearchResults<blokus::GameState>;

  static void dump(const LocalPolicyArray& action_policy, const SearchResults& results);
};

}  // namespace mcts

#include <games/blokus/inl/GameState.inl>
