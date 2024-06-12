#include <games/connect4/players/PerfectPlayer.hpp>

#include <util/BitSet.hpp>
#include <util/BoostUtil.hpp>
#include <util/Exception.hpp>
#include <util/RepoUtil.hpp>
#include <util/StringUtil.hpp>

#include <boost/dll.hpp>

namespace c4 {

PerfectOracle::oracle_vec_t PerfectOracle::oracles_;
std::mutex PerfectOracle::static_mutex_;

std::string PerfectOracle::QueryResult::get_overlay() const {
  char chars[kNumColumns];

  for (int i = 0; i < kNumColumns; ++i) {
    int score = scores[i];
    if (score < 0) {
      chars[i] = ' ';
    } else if (score == 0) {
      chars[i] = '0';
    } else {
      chars[i] = '+';
    }
  }
  return util::create_string(" %c %c %c %c %c %c %c", chars[0], chars[1], chars[2], chars[3],
                             chars[4], chars[5], chars[6]);
}

inline PerfectOracle::QueryResult PerfectOracle::query(MoveHistory& history) {
  std::string s;

  {
    std::lock_guard lock(mutex_);
    history.write(in_);
    std::getline(out_, s);
  }
  auto tokens = util::split(s);

  QueryResult result;
  for (int j = 0; j < kNumColumns; ++j) {
    int raw_score = std::stoi(tokens[tokens.size() - kNumColumns + j]);
    if (raw_score == QueryResult::kIllegalMoveScore) {
      result.scores[j] = QueryResult::kIllegalMoveScore;
    } else if (raw_score < 0) {
      result.scores[j] = -22 + (history.length() + 1) / 2 - raw_score;
    } else if (raw_score > 0) {
      result.scores[j] = 22 - history.length() / 2 - raw_score;
    } else {
      result.scores[j] = 0;
    }
  }

  int max_score = result.scores.maxCoeff();
  if (max_score > 0) {
    // set best_score to the positive score closest to 0
    result.best_score = max_score;
    for (int j = 0; j < kNumColumns; ++j) {
      if (result.scores[j] > 0 && result.scores[j] < result.best_score) {
        result.best_score = result.scores[j];
      }
    }
  } else if (max_score < 0) {
    // set best_score to the most negative non-illegal score
    result.best_score = 0;
    for (int j = 0; j < kNumColumns; ++j) {
      int score = result.scores[j];
      if (score < result.best_score && score != QueryResult::kIllegalMoveScore) {
        result.best_score = result.scores[j];
      }
    }
  } else {
    result.best_score = 0;
  }
  return result;
}

core::ActionResponse PerfectPlayer::get_action_response(
    const FullState& state, const ActionMask& valid_actions) {
  auto result = oracle_->query(move_history_);

  core::ActionResponse response;

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
    c4::Game::IO::print_state(state.base());
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
