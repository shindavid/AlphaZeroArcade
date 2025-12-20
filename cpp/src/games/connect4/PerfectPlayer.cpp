#include "games/connect4/players/PerfectPlayer.hpp"

#include "core/BasicTypes.hpp"

#include <iostream>

namespace c4 {

PerfectPlayer::ActionResponse PerfectPlayer::get_action_response(const ActionRequest& request) {
  if (request.aux) {
    return ActionResponse::make_move(request.aux - 1);
  }

  PerfectOracle* oracle = oracle_pool_->get_oracle(request.notification_unit);
  if (!oracle) {
    return ActionResponse::yield();
  }
  PerfectOracle::QueryResult result = oracle->query(move_history_);
  oracle_pool_->release_oracle(oracle);

  ActionResponse response = ActionResponse::make_move(core::kNullAction);

  ActionMask candidates;

  // first add clearly winning moves
  for (int j = 0; j < kNumColumns; ++j) {
    if (result.scores[j] > 0 && result.scores[j] <= params_.strength) {
      candidates[j] = 1;
    }
  }

  // if no known winning moves, then add all draws/uncertain moves
  bool known_win = candidates.any();
  response.victory_guarantee = known_win;
  if (!known_win) {
    for (int j = 0; j < kNumColumns; ++j) {
      int score = result.scores[j];
      if (score == PerfectOracle::QueryResult::kIllegalMoveScore) {
        continue;
      }
      candidates[j] = abs(score) > params_.strength || score == 0;
    }
  }

  // if no candidates, then everything is a certain loss. Choose randomly among slowest losses.
  if (!candidates.any()) {
    for (int j = 0; j < kNumColumns; ++j) {
      candidates[j] = result.scores[j] == result.best_score;
    }
  }

  if (params_.verbose) {
    std::cout << "get_action_response()" << std::endl;
    c4::Game::IO::print_state(std::cout, request.state);
    std::cout << "scores: " << result.scores.transpose() << std::endl;
    std::cout << "best_score: " << result.best_score << std::endl;
    std::cout << "my_strength: " << params_.strength << std::endl;
    std::cout << "candidates:";
    for (int j : candidates.on_indices()) {
      std::cout << " " << (j + 1);
    }
    std::cout << std::endl;
  }

  response.action = candidates.choose_random_on_index();
  response.set_aux(response.action + 1);
  return response;
}

}  // namespace c4
