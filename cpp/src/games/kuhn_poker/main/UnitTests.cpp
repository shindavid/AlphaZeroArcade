#include "core/GameServer.hpp"
#include "games/kuhn_poker/Game.hpp"
#include "generic_players/RandomPlayer.hpp"
#include "generic_players/RandomPlayerGenerator.hpp"
#include "util/GTestUtil.hpp"
#include "util/Random.hpp"
#include "util/RepoUtil.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using Game = kuhn_poker::Game;
using State = Game::State;
using InfoSet = Game::InfoSet;
using Move = Game::Move;
using MoveSet = Game::MoveSet;
using Rules = Game::Rules;
using IO = Game::IO;
using GameOutcome = Game::Types::GameOutcome;

// ============================================================================
// Basic rules tests
// ============================================================================

TEST(KuhnPokerRules, InitialState) {
  State state;
  Rules::init_state(state);
  EXPECT_TRUE(Rules::is_chance_state(state));
  EXPECT_EQ(state.phase, kuhn_poker::kDealPhase);
  EXPECT_EQ(state.cards[0], -1);
  EXPECT_EQ(state.cards[1], -1);
}

TEST(KuhnPokerRules, Deal) {
  State state;
  Rules::init_state(state);

  // Deal index 0: J to P0, Q to P1
  Rules::apply(state, Move(0, kuhn_poker::kDealPhase));
  EXPECT_FALSE(Rules::is_chance_state(state));
  EXPECT_EQ(state.cards[0], 0);  // Jack
  EXPECT_EQ(state.cards[1], 1);  // Queen
  EXPECT_EQ(Rules::get_current_player(state), 0);
}

TEST(KuhnPokerRules, CheckCheck) {
  State state;
  Rules::init_state(state);
  Rules::apply(state, Move(0, kuhn_poker::kDealPhase));  // J vs Q

  Rules::apply(state, Move(kuhn_poker::kCheck, kuhn_poker::kBettingPhase));
  EXPECT_FALSE(Rules::analyze(state).is_terminal());

  Rules::apply(state, Move(kuhn_poker::kCheck, kuhn_poker::kBettingPhase));
  EXPECT_TRUE(Rules::analyze(state).is_terminal());

  // Queen (P1) beats Jack (P0)
  auto outcome = Rules::analyze(state).outcome();
  EXPECT_EQ(outcome[0].score, -1.0f);
  EXPECT_EQ(outcome[1].score, 1.0f);
}

TEST(KuhnPokerRules, BetFold) {
  State state;
  Rules::init_state(state);
  Rules::apply(state, Move(0, kuhn_poker::kDealPhase));  // J vs Q

  Rules::apply(state, Move(kuhn_poker::kBet, kuhn_poker::kBettingPhase));   // P0 bets
  Rules::apply(state, Move(kuhn_poker::kFold, kuhn_poker::kBettingPhase));  // P1 folds

  EXPECT_TRUE(Rules::analyze(state).is_terminal());
  auto outcome = Rules::analyze(state).outcome();
  // P1 folded, so P0 wins regardless of cards
  EXPECT_EQ(outcome[0].score, 1.0f);
  EXPECT_EQ(outcome[1].score, -1.0f);
}

TEST(KuhnPokerRules, BetCall) {
  State state;
  Rules::init_state(state);
  Rules::apply(state, Move(0, kuhn_poker::kDealPhase));  // J vs Q

  Rules::apply(state, Move(kuhn_poker::kBet, kuhn_poker::kBettingPhase));
  Rules::apply(state, Move(kuhn_poker::kCall, kuhn_poker::kBettingPhase));

  EXPECT_TRUE(Rules::analyze(state).is_terminal());
  auto outcome = Rules::analyze(state).outcome();
  // Queen beats Jack → P1 wins
  EXPECT_EQ(outcome[0].score, -2.0f);
  EXPECT_EQ(outcome[1].score, 2.0f);
}

TEST(KuhnPokerRules, CheckBetCall) {
  State state;
  Rules::init_state(state);
  Rules::apply(state, Move(4, kuhn_poker::kDealPhase));  // K vs J

  Rules::apply(state, Move(kuhn_poker::kCheck, kuhn_poker::kBettingPhase));  // P0 checks
  Rules::apply(state, Move(kuhn_poker::kBet, kuhn_poker::kBettingPhase));    // P1 bets
  Rules::apply(state, Move(kuhn_poker::kCall, kuhn_poker::kBettingPhase));   // P0 calls

  EXPECT_TRUE(Rules::analyze(state).is_terminal());
  auto outcome = Rules::analyze(state).outcome();
  // King (P0) beats Jack (P1)
  EXPECT_EQ(outcome[0].score, 2.0f);
  EXPECT_EQ(outcome[1].score, -2.0f);
}

TEST(KuhnPokerRules, CheckBetFold) {
  State state;
  Rules::init_state(state);
  Rules::apply(state, Move(0, kuhn_poker::kDealPhase));  // J vs Q

  Rules::apply(state, Move(kuhn_poker::kCheck, kuhn_poker::kBettingPhase));  // P0 checks
  Rules::apply(state, Move(kuhn_poker::kBet, kuhn_poker::kBettingPhase));    // P1 bets
  Rules::apply(state, Move(kuhn_poker::kFold, kuhn_poker::kBettingPhase));   // P0 folds

  EXPECT_TRUE(Rules::analyze(state).is_terminal());
  auto outcome = Rules::analyze(state).outcome();
  // P0 folded, P1 wins
  EXPECT_EQ(outcome[0].score, -1.0f);
  EXPECT_EQ(outcome[1].score, 1.0f);
}

TEST(KuhnPokerRules, LegalMovesAfterDeal) {
  State state;
  Rules::init_state(state);
  Rules::apply(state, Move(0, kuhn_poker::kDealPhase));

  auto result = Rules::analyze(state);
  EXPECT_FALSE(result.is_terminal());
  auto moves = result.valid_moves();
  EXPECT_EQ(moves.size(), 2);
  EXPECT_TRUE(moves.contains(Move(kuhn_poker::kCheck, kuhn_poker::kBettingPhase)));
  EXPECT_TRUE(moves.contains(Move(kuhn_poker::kBet, kuhn_poker::kBettingPhase)));
}

TEST(KuhnPokerRules, LegalMovesAfterBet) {
  State state;
  Rules::init_state(state);
  Rules::apply(state, Move(0, kuhn_poker::kDealPhase));
  Rules::apply(state, Move(kuhn_poker::kBet, kuhn_poker::kBettingPhase));

  auto result = Rules::analyze(state);
  EXPECT_FALSE(result.is_terminal());
  auto moves = result.valid_moves();
  EXPECT_EQ(moves.size(), 2);
  EXPECT_TRUE(moves.contains(Move(kuhn_poker::kFold, kuhn_poker::kBettingPhase)));
  EXPECT_TRUE(moves.contains(Move(kuhn_poker::kCall, kuhn_poker::kBettingPhase)));
}

// ============================================================================
// InfoSet tests
// ============================================================================

TEST(KuhnPokerInfoSet, HidesOpponentCard) {
  State state;
  Rules::init_state(state);
  Rules::apply(state, Move(0, kuhn_poker::kDealPhase));  // J vs Q

  InfoSet info0 = Rules::state_to_info_set(state, 0);
  InfoSet info1 = Rules::state_to_info_set(state, 1);

  // Each player sees only their own card
  EXPECT_EQ(info0.my_card, 0);  // Jack
  EXPECT_EQ(info1.my_card, 1);  // Queen

  // Both see the same public betting info
  EXPECT_EQ(info0.num_actions, 0);
  EXPECT_EQ(info1.num_actions, 0);
  EXPECT_EQ(info0.phase, kuhn_poker::kBettingPhase);
}

TEST(KuhnPokerInfoSet, PreservesBettingHistory) {
  State state;
  Rules::init_state(state);
  Rules::apply(state, Move(0, kuhn_poker::kDealPhase));
  Rules::apply(state, Move(kuhn_poker::kCheck, kuhn_poker::kBettingPhase));

  InfoSet info0 = Rules::state_to_info_set(state, 0);
  EXPECT_EQ(info0.num_actions, 1);
  EXPECT_EQ(info0.actions[0], kuhn_poker::kCheck);
}

TEST(KuhnPokerInfoSet, SameInfoSetDifferentDeals) {
  // Two different states where P0 has Jack and P0 checked
  State state_jq, state_jk;
  Rules::init_state(state_jq);
  Rules::init_state(state_jk);
  Rules::apply(state_jq, Move(0, kuhn_poker::kDealPhase));  // J vs Q
  Rules::apply(state_jk, Move(1, kuhn_poker::kDealPhase));  // J vs K

  Rules::apply(state_jq, Move(kuhn_poker::kCheck, kuhn_poker::kBettingPhase));
  Rules::apply(state_jk, Move(kuhn_poker::kCheck, kuhn_poker::kBettingPhase));

  // P0's info set should be identical (same card, same history)
  InfoSet info0_jq = Rules::state_to_info_set(state_jq, 0);
  InfoSet info0_jk = Rules::state_to_info_set(state_jk, 0);
  EXPECT_EQ(info0_jq, info0_jk);

  // But P1's info sets should differ (different cards)
  InfoSet info1_jq = Rules::state_to_info_set(state_jq, 1);
  InfoSet info1_jk = Rules::state_to_info_set(state_jk, 1);
  EXPECT_NE(info1_jq, info1_jk);
}

// ============================================================================
// IO tests
// ============================================================================

TEST(KuhnPokerIO, CompactRepr) {
  State state;
  Rules::init_state(state);
  EXPECT_EQ(IO::compact_state_repr(state), "deal");

  Rules::apply(state, Move(0, kuhn_poker::kDealPhase));
  EXPECT_EQ(IO::compact_state_repr(state), "JQ");

  Rules::apply(state, Move(kuhn_poker::kCheck, kuhn_poker::kBettingPhase));
  EXPECT_EQ(IO::compact_state_repr(state), "JQ-check");

  Rules::apply(state, Move(kuhn_poker::kBet, kuhn_poker::kBettingPhase));
  EXPECT_EQ(IO::compact_state_repr(state), "JQ-check-bet");
}

TEST(KuhnPokerMove, RoundTrip) {
  State state;
  Rules::init_state(state);

  // Deal moves
  InfoSet info0 = Rules::state_to_info_set(state, 0);
  for (int i = 0; i < kuhn_poker::kNumDeals; ++i) {
    Move m(i, kuhn_poker::kDealPhase);
    EXPECT_EQ(Move::from_str(info0, m.to_str()), m);
  }

  // Betting moves
  Rules::apply(state, Move(0, kuhn_poker::kDealPhase));
  InfoSet info1 = Rules::state_to_info_set(state, 0);
  for (int i = 0; i < kuhn_poker::kNumBettingActions; ++i) {
    Move m(i, kuhn_poker::kBettingPhase);
    EXPECT_EQ(Move::from_str(info1, m.to_str()), m);
  }
}

// ============================================================================
// GameServer + RandomPlayer integration test
// ============================================================================

class KuhnPokerGameServerTest : public testing::Test {
 protected:
  using GameServer = core::GameServer<Game>;
  using GameServerParams = GameServer::Params;
  using results_array_t = GameServer::results_array_t;

  void SetUp() override {
    util::Random::set_seed(0);
    core::PerfStatsRegistry::clear();
  }

  results_array_t run_games(int num_games) {
    GameServerParams params;
    params.num_games = num_games;
    params.num_game_threads = 1;
    params.deterministic_mode = true;

    GameServer server(params);
    auto* gen0 = new generic::RandomPlayerGenerator<Game>(&server);
    auto* gen1 = new generic::RandomPlayerGenerator<Game>(&server);
    gen0->set_base_seed(42);
    gen1->set_base_seed(123);
    server.register_player(-1, gen0);
    server.register_player(-1, gen1);
    server.run();
    return server.get_results();
  }
};

TEST_F(KuhnPokerGameServerTest, SingleGame) {
  auto results = run_games(1);

  // Verify that scores are zero-sum
  float total = results[0].total + results[1].total;
  EXPECT_EQ(total, 0.0f);
}

TEST_F(KuhnPokerGameServerTest, MultipleGames) {
  auto results = run_games(100);

  // Over 100 games, both players should have some wins
  EXPECT_GT(results[0].win, 0);
  EXPECT_GT(results[1].win, 0);

  // Scores are zero-sum
  float total = results[0].total + results[1].total;
  EXPECT_FLOAT_EQ(total, 0.0f);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
