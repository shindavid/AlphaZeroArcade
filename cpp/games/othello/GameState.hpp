#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <tuple>

#include <boost/functional/hash.hpp>
#include <torch/torch.h>

#include <core/AbstractPlayer.hpp>
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
  static constexpr int kNumGlobalActions = othello::kNumGlobalActions;
  static constexpr int kMaxNumLocalActions = othello::kMaxNumLocalActions;
  static constexpr int kTypicalNumMovesPerGame = othello::kTypicalNumMovesPerGame;

  using GameStateTypes = core::GameStateTypes<GameState>;
  using ActionMask = GameStateTypes::ActionMask;
  using player_name_array_t = GameStateTypes::player_name_array_t;
  using ValueArray = GameStateTypes::ValueArray;
  using LocalPolicyArray = GameStateTypes::LocalPolicyArray;
  using GameOutcome = GameStateTypes::GameOutcome;

  core::seat_index_t get_current_player() const { return cur_player_; }
  GameOutcome apply_move(core::action_t action);
  ActionMask get_valid_actions() const;

  template<eigen_util::FixedTensorConcept InputTensor> void tensorize(InputTensor&) const;
  void dump(core::action_t last_action=-1, const player_name_array_t* player_names=nullptr) const;
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
