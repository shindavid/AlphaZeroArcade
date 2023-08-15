#include <games/tictactoe/GameState.hpp>

inline std::size_t std::hash<tictactoe::GameState>::operator()(const tictactoe::GameState& state) const {
  return state.hash();
}

namespace tictactoe {

inline core::seat_index_t GameState::get_current_player() const {
  return std::popcount(full_mask_) % 2;
}

inline GameState::GameOutcome GameState::apply_move(core::action_t action) {
  core::seat_index_t current_player = get_current_player();

  mask_t piece_mask = mask_t(1) << action;
  cur_player_mask_ ^= full_mask_;
  full_mask_ |= piece_mask;

  bool win = false;

  mask_t masks[] = {
    make_mask(0, 1, 2),
    make_mask(3, 4, 5),
    make_mask(6, 7, 8),
    make_mask(0, 3, 6),
    make_mask(1, 4, 7),
    make_mask(2, 5, 8),
    make_mask(0, 4, 8),
    make_mask(2, 4, 6)
  };

  mask_t updated_mask = full_mask_ ^ cur_player_mask_;
  for (mask_t mask : masks) {
    if ((mask & updated_mask) == mask) {
      win = true;
      break;
    }
  }

  GameOutcome outcome;
  outcome.setZero();
  if (win) {
    outcome(current_player) = 1.0;
  } else if (std::popcount(full_mask_) == kNumCells) {
    outcome(0) = 0.5;
    outcome(1) = 0.5;
  }

  return outcome;
}

inline GameState::ActionMask GameState::get_valid_actions() const {
  uint64_t u = full_mask_;
  ActionMask& mask = reinterpret_cast<ActionMask&>(u);
  return ~mask;
}

template<eigen_util::FixedTensorConcept InputTensor> void GameState::tensorize(InputTensor& tensor) const {
  mask_t opp_player_mask = full_mask_ ^ cur_player_mask_;
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      int index = row * kBoardDimension + col;
      bool occupied_by_cur_player = (mask_t(1) << index) & cur_player_mask_;
      tensor(0, row, col) = occupied_by_cur_player;
    }
  }
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      int index = row * kBoardDimension + col;
      bool occupied_by_opp_player = (mask_t(1) << index) & opp_player_mask;
      tensor(1, row, col) = occupied_by_opp_player;
    }
  }
}

inline void GameState::dump(core::action_t last_action, const player_name_array_t* player_names) const {
  auto cp = get_current_player();
  mask_t opp_player_mask = full_mask_ ^ cur_player_mask_;
  mask_t o_mask = (cp == kO) ? cur_player_mask_ : opp_player_mask;
  mask_t x_mask = (cp == kX) ? cur_player_mask_ : opp_player_mask;

  char text[] =  "6 7 8  | | | |\n"
                 "3 4 5  | | | |\n"
                 "0 1 2  | | | |\n";

  int offset_table[] = {38, 40, 42, 23, 25, 27, 8, 10, 12};
  for (int i = 0; i < kNumCells; ++i) {
    int offset = offset_table[i];
    if (o_mask & (mask_t(1) << i)) {
      text[offset] = 'O';
    } else if (x_mask & (mask_t(1) << i)) {
      text[offset] = 'X';
    }
  }

  printf("%s\n", text);

  if (player_names) {
    printf("X: %s\n", (*player_names)[kX].c_str());
    printf("O: %s\n\n", (*player_names)[kO].c_str());
  }
  std::cout.flush();
}

}  // namespace tictactoe

namespace mcts {

inline void SearchResultsDumper<tictactoe::GameState>::dump(
    const LocalPolicyArray &action_policy, const SearchResults &results)
{
  const auto& valid_actions = results.valid_actions;
  const auto& mcts_counts = results.counts;
  const auto& net_policy = results.policy_prior;
  const auto& win_rates = results.win_rates;
  const auto& net_value = results.value_prior;

  assert(net_policy.size() == (int)valid_actions.count());

  printf("X: %6.3f%% -> %6.3f%%\n", 100 * net_value(tictactoe::kX), 100 * win_rates(tictactoe::kO));
  printf("O: %6.3f%% -> %6.3f%%\n", 100 * net_value(tictactoe::kO), 100 * win_rates(tictactoe::kX));
  printf("\n");

  int num_actions = valid_actions.count();
  printf("%4s %8s %8s %8s\n", "Move", "Net", "Count", "MCTS");
  for (core::action_t action : bitset_util::on_indices(valid_actions)) {
    float count = mcts_counts.data()[action];
    auto action_p = action_policy(action);
    auto net_p = net_policy(action);
    printf("   %d %8.3f %8.3f %8.3f\n", action, net_p, count, action_p);
  }
  for (int i = num_actions; i < tictactoe::kNumCells; ++i) {
    printf("\n");
  }
}

}  // namespace mcts
