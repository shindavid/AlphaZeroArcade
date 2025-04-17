#include <games/connect4/players/PerfectPlayer.hpp>

#include <util/BitSet.hpp>
#include <util/BoostUtil.hpp>
#include <util/Exception.hpp>
#include <util/RepoUtil.hpp>
#include <util/StringUtil.hpp>

#include <boost/dll.hpp>

namespace c4 {

PerfectPlayer::ActionResponse PerfectPlayer::get_action_response(const ActionRequest& request) {
  PerfectOracle::QueryResult result;
  if (oracle_ == nullptr) {
    oracle_ = oracle_pool_->get_oracle();
    if (oracle_) {
      oracle_->async_query(move_history_);
    }
    return ActionResponse::make_yield();
  } else {
    if (!oracle_->async_load(result)) {
      // oracle_ is still working on it
      return ActionResponse::make_yield();
    }
  }

  oracle_pool_->release_oracle(oracle_);
  oracle_ = nullptr;

  ActionResponse response;

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
    for (int j : bitset_util::on_indices(candidates)) {
      std::cout << " " << (j + 1);
    }
    std::cout << std::endl;
  }

  response.action = bitset_util::choose_random_on_index(candidates);
  return response;
}

}  // namespace c4
