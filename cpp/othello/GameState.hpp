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
#include <othello/Constants.hpp>
#include <util/EigenUtil.hpp>

namespace othello { class GameState; }

template <>
struct std::hash<othello::GameState> {
  std::size_t operator()(const othello::GameState& state) const;
};

namespace othello {

/*
 * Bit order encoding for the board:
 *
 * 56 57 58 59 60 61 62 63
 * 48 49 50 51 52 53 54 55
 * 40 41 42 43 44 45 46 47
 * 32 33 34 35 36 37 38 39
 * 24 25 26 27 28 29 30 31
 * 16 17 18 19 20 21 22 23
 *  8  9 10 11 12 13 14 15
 *  0  1  2  3  4  5  6  7
 *
 * The algorithms for manipulating the board are lifted from:
 *
 * https://github.com/abulmo/edax-reversi
 *
 * For human-readable notation purposes, we adopt chess-notation:
 *
 * A8 B8 C8 D8 E8 F8 G8 H8
 * A7 B7 C7 D7 E7 F7 G7 H7
 * A6 B6 C6 D6 E6 F6 G6 H6
 * A5 B5 C5 D5 E5 F5 G5 H5
 * A4 B4 C4 D4 E4 F4 G4 H4
 * A3 B3 C3 D3 E3 F3 G3 H3
 * A2 B2 C2 D2 E2 F2 G2 H2
 * A1 B1 C1 D1 E1 F1 G1 H1
 *
 */
class GameState {
public:
  static constexpr int kNumPlayers = othello::kNumPlayers;
  static constexpr int kNumGlobalActions = othello::kNumGlobalActions;
  static constexpr int kMaxNumLocalActions = othello::kNumGlobalActions;
  static constexpr int kTypicalNumMovesPerGame = othello::kTypicalNumMovesPerGame;

  using GameStateTypes = common::GameStateTypes<GameState>;
  using ActionMask = GameStateTypes::ActionMask;
  using player_name_array_t = GameStateTypes::player_name_array_t;
  using ValueProbDistr = GameStateTypes::ValueProbDistr;
  using MctsResults = common::MctsResults<GameState>;
  using LocalPolicyProbDistr = GameStateTypes::LocalPolicyProbDistr;
  using GameOutcome = GameStateTypes::GameOutcome;

  static size_t serialize_action(char* buffer, size_t buffer_size, common::action_index_t action);
  static void deserialize_action(const char* buffer, common::action_index_t* action);

  size_t serialize_action_prompt(char* buffer, size_t buffer_size, const ActionMask& valid_actions) const { return 0; }
  void deserialize_action_prompt(const char* buffer, ActionMask* valid_actions) const { *valid_actions = get_valid_actions(); }

  size_t serialize_state_change(char* buffer, size_t buffer_size, common::seat_index_t seat,
                                common::action_index_t action) const;
  void deserialize_state_change(const char* buffer, common::seat_index_t* seat, common::action_index_t* action);

  size_t serialize_game_end(char* buffer, size_t buffer_size, const GameOutcome& outcome) const;
  void deserialize_game_end(const char* buffer, GameOutcome* outcome);

  common::seat_index_t get_current_player() const { return cur_player_; }
  GameOutcome apply_move(common::action_index_t action);
  ActionMask get_valid_actions() const;

  template<eigen_util::FixedTensorConcept InputSlab> void tensorize(InputSlab&) const;
  void dump(common::action_index_t last_action=-1, const player_name_array_t* player_names=nullptr) const;
  bool operator==(const GameState& other) const;
  std::size_t hash() const { return boost::hash_range(&full_mask_, (&full_mask_) + 2); }

  static common::action_index_t prompt_for_action();
  static void dump_mcts_output(const ValueProbDistr& mcts_value, const LocalPolicyProbDistr& mcts_policy,
                               const MctsResults& results);

private:
  void row_dump(row_t row, column_t blink_column) const;
  static mask_t get_moves(mask_t P, mask_t O);
  static mask_t get_some_moves(mask_t P, mask_t mask, int dir);

  mask_t full_mask_ = 0;  // spaces occupied by either player
  mask_t cur_player_mask_ = 0;  // spaces occupied by current player
  common::seat_index_t cur_player_ = kBlack;
};

static_assert(common::GameStateConcept<othello::GameState>);

using Player = common::AbstractPlayer<GameState>;

}  // namespace c4

#include <othello/inl/GameState.inl>
