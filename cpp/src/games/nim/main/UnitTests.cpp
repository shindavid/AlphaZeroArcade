#include "games/nim/Bindings.hpp"
#include "games/nim/Game.hpp"
#include "games/nim/PolicyEncoding.hpp"
#include "util/GTestUtil.hpp"

#include <gtest/gtest.h>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using Game = nim::Game;
using State = Game::State;
using PolicyTensor = nim::PolicyEncoding::Tensor;
using IO = Game::IO;
using Rules = Game::Rules;
using SymmetryGroup = groups::TrivialGroup;
using GameOutcome = Game::GameOutcome;
using InputEncoder = nim::InputEncoder;
using Move = Game::Types::Move;

TEST(Analyze, FromInitState) {
  State state;
  Rules::init_state(state);

  auto valid_masks = Rules::analyze(state).valid_moves();
  EXPECT_TRUE(valid_masks.size() == 3);
}

TEST(NimGameTest, InitialState) {
  State state;
  Rules::init_state(state);

  EXPECT_EQ(Rules::get_current_player(state), 0);
  EXPECT_EQ(state.stones_left, 21);  // Assuming the game starts with 21 stones
}

TEST(NimGameTest, MakeMove) {
  State state;
  Rules::init_state(state);
  Rules::apply(state, nim::Move(nim::kTake3));

  EXPECT_EQ(state.stones_left, 18);
  EXPECT_EQ(Rules::get_current_player(state), 1);
}

TEST(NimGameTest, Player0Wins) {
  State state;
  Rules::init_state(state);
  std::vector<int> actions = {nim::kTake3, nim::kTake3, nim::kTake3, nim::kTake3,
                              nim::kTake3, nim::kTake3, nim::kTake3};

  for (int action : actions) {
    Rules::apply(state, nim::Move(action));
  }

  auto result = Rules::analyze(state);
  bool terminal = result.is_terminal();
  GameOutcome outcome = result.outcome();

  EXPECT_TRUE(terminal);
  EXPECT_EQ(outcome[0].share, 1);
}

TEST(NimGameTest, Player1Wins) {
  State state;
  Rules::init_state(state);
  std::vector<int> actions = {nim::kTake3, nim::kTake3, nim::kTake3, nim::kTake3,
                              nim::kTake3, nim::kTake3, nim::kTake1, nim::kTake2};

  for (int action : actions) {
    Rules::apply(state, nim::Move(action));
  }

  auto result = Rules::analyze(state);
  bool terminal = result.is_terminal();
  GameOutcome outcome = result.outcome();

  EXPECT_TRUE(terminal);
  EXPECT_EQ(outcome[1].share, 1);
}

TEST(NimGameTest, InvalidMove) {
  State state;
  Rules::init_state(state);
  EXPECT_THROW(Rules::apply(state, nim::Move(-1)), std::invalid_argument);
  EXPECT_THROW(Rules::apply(state, nim::Move(3)), std::invalid_argument);
}

TEST(NimGameTest, encode) {
  State state;
  Rules::init_state(state);
  Rules::apply(state, nim::Move(1));  // Player 0
  Rules::apply(state, nim::Move(0));  // Player 1

  InputEncoder input_encoder;
  input_encoder.update(state);

  InputEncoder::Tensor tensor = input_encoder.encode();
  float expectedValues[] = {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  for (int i = 0; i < tensor.size(); i++) {
    EXPECT_EQ(tensor.data()[i], expectedValues[i]);
  }
}

TEST(Move, RoundTrip) {
  State state;
  Rules::init_state(state);

  // take1 serializes as "1", take2 as "2", take3 as "3"
  EXPECT_EQ(nim::Move(nim::kTake1).to_str(), "1");
  EXPECT_EQ(nim::Move(nim::kTake2).to_str(), "2");
  EXPECT_EQ(nim::Move(nim::kTake3).to_str(), "3");

  // from_str inverts to_str
  EXPECT_EQ(Move::from_str(state, "1"), nim::Move(nim::kTake1));
  EXPECT_EQ(Move::from_str(state, "2"), nim::Move(nim::kTake2));
  EXPECT_EQ(Move::from_str(state, "3"), nim::Move(nim::kTake3));

  // round-trip for each valid move
  for (int i = nim::kTake1; i <= nim::kTake3; ++i) {
    nim::Move m(i);
    EXPECT_EQ(Move::from_str(state, m.to_str()), m) << "round-trip failed for move " << i;
  }
}

TEST(NimGameTest, NearTerminal_MultipleValidMoves) {
  // With exactly 3 stones left, all three moves are valid
  State state;
  Rules::init_state(state);
  state.stones_left = 3;

  auto result = Rules::analyze(state);
  EXPECT_FALSE(result.is_terminal());
  EXPECT_EQ(result.valid_moves().size(), 3);
  EXPECT_TRUE(result.valid_moves().contains(nim::Move(nim::kTake1)));
  EXPECT_TRUE(result.valid_moves().contains(nim::Move(nim::kTake2)));
  EXPECT_TRUE(result.valid_moves().contains(nim::Move(nim::kTake3)));
}

TEST(NimGameTest, NearTerminal_OnlyOneOption) {
  // With exactly 1 stone left, only kTake1 is valid
  State state;
  Rules::init_state(state);
  state.stones_left = 1;

  auto result = Rules::analyze(state);
  EXPECT_FALSE(result.is_terminal());
  EXPECT_EQ(result.valid_moves().size(), 1);
  EXPECT_TRUE(result.valid_moves().contains(nim::Move(nim::kTake1)));
  EXPECT_FALSE(result.valid_moves().contains(nim::Move(nim::kTake2)));
  EXPECT_FALSE(result.valid_moves().contains(nim::Move(nim::kTake3)));
}

TEST(NimGameTest, NearTerminal_TakeAllEndsGame) {
  // With 3 stones left, taking all 3 terminates the game
  State state;
  Rules::init_state(state);
  state.stones_left = 3;

  Rules::apply(state, nim::Move(nim::kTake3));

  auto result = Rules::analyze(state);
  EXPECT_TRUE(result.is_terminal());
  // Player 0 took the last stones → player 0 wins
  EXPECT_EQ(result.outcome()[0].share, 1.0f);
}

int main(int argc, char **argv) { return launch_gtest(argc, argv); }
