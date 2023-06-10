#include <connect4/players/PerfectPlayer.hpp>

#include <util/BitSet.hpp>
#include <util/BoostUtil.hpp>
#include <util/Config.hpp>
#include <util/Exception.hpp>
#include <util/RepoUtil.hpp>
#include <util/StringUtil.hpp>

namespace c4 {

inline PerfectOracle::MoveHistory::MoveHistory() : char_pointer_(chars_) {}

inline PerfectOracle::MoveHistory::MoveHistory(const MoveHistory& history) {
  memcpy(chars_, history.chars_, sizeof(chars_));
  char_pointer_ = chars_ + (history.char_pointer_ - history.chars_);
}

inline void PerfectOracle::MoveHistory::reset() {
  char_pointer_ = chars_;
  *char_pointer_ = 0;
}

inline void PerfectOracle::MoveHistory::append(common::action_index_t move) {
  *(char_pointer_++) = char(int('1') + move);  // connect4 program uses 1-indexing
}

inline std::string PerfectOracle::MoveHistory::to_string() const {
  return std::string(chars_, char_pointer_ - chars_);
}

inline void PerfectOracle::MoveHistory::write(boost::process::opstream& in) {
  *char_pointer_ = '\n';
  in.write(chars_, char_pointer_ - chars_ + 1);
  in.flush();
}

inline std::string PerfectOracle::QueryResult::get_overlay() const {
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
  return util::create_string(" %c %c %c %c %c %c %c",
                             chars[0], chars[1], chars[2], chars[3],
                             chars[4], chars[5], chars[6]);
}

inline PerfectOracle::PerfectOracle() {
  std::string c4_solver_dir_str = util::Config::instance()->get("c4.solver_dir", "");

  if (c4_solver_dir_str.empty()) {
    throw util::Exception("c4 solver dir not specified! Please add 'c4.solver_dir' entry in $REPO_ROOT/%s",
                          util::Config::kFilename);
  }
  boost::filesystem::path c4_solver_dir(c4_solver_dir_str);
  if (!boost::filesystem::is_directory(c4_solver_dir)) {
    throw util::Exception("Directory does not exist: %s", c4_solver_dir.c_str());
  }
  boost::filesystem::path c4_solver_bin = c4_solver_dir / "c4solver";
  boost::filesystem::path c4_solver_book = c4_solver_dir / "7x6.book";
  for (const auto& path : {c4_solver_bin, c4_solver_book}) {
    if (!boost::filesystem::is_regular_file(path)) {
      throw util::Exception("File does not exist: %s", path.c_str());
    }
  }

  namespace bp = boost::process;
  std::string c4_cmd = util::create_string("%s -b %s -a", c4_solver_bin.c_str(), c4_solver_book.c_str());
  proc_ = new bp::child(c4_cmd, bp::std_out > out_, bp::std_err > bp::null, bp::std_in < in_);
}

inline PerfectOracle::~PerfectOracle() {
  delete proc_;
}

inline PerfectOracle::QueryResult PerfectOracle::query(MoveHistory &history) {
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

inline auto PerfectPlayer::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("c4::PerfectPlayer options");
  return desc
      .template add_option<"strength", 's'>
          (po::value<int>(&strength)->default_value(strength),
           "strength (0-21). The last s moves are played perfectly, the others randomly. 0 is random, 21 is perfect.")
      .template add_option<"verbose", 'v'>(
          po::bool_switch(&verbose)->default_value(verbose), "mcts player verbose mode")
      ;
}

inline PerfectPlayer::PerfectPlayer(const Params& params) : params_(params) {
  util::clean_assert(params_.strength >= 0 && params_.strength <= 21, "strength must be in [0, 21]");
}

inline void PerfectPlayer::start_game() {
  move_history_.reset();
}

inline void PerfectPlayer::receive_state_change(
    common::seat_index_t, const GameState&, common::action_index_t action)
{
  move_history_.append(action);
}

inline common::action_index_t PerfectPlayer::get_action(const GameState& state, const ActionMask& valid_actions) {
  auto result = oracle_.query(move_history_);

  ActionMask candidates;

  // first add clearly winning moves
  for (int j = 0; j < kNumColumns; ++j) {
    if (result.scores[j] > 0 && result.scores[j] <= params_.strength) {
      candidates.set(j);
    }
  }

  // if no known winning moves, then add all draws/uncertain moves
  bool known_win = candidates.any();
  if (!known_win) {
    for (int j = 0; j < kNumColumns; ++j) {
      int score = result.scores[j];
      if (score == PerfectOracle::QueryResult::kIllegalMoveScore) {
        continue;
      }
      candidates.set(j, abs(score) > params_.strength || score == 0);
    }
  }

  // if no candidates, then everything is a certain loss. Choose randomly among slowest losses.
  if (!candidates.any()) {
    for (int j = 0; j < kNumColumns; ++j) {
      candidates.set(j, result.scores[j] == result.best_score);
    }
  }

  if (params_.verbose) {
    std::cout << "get_action()" << std::endl;
    state.dump();
    std::cout << "scores: " << result.scores.transpose() << std::endl;
    std::cout << "best_score: " << result.best_score << std::endl;
    std::cout << "my_strength: " << params_.strength << std::endl;
    std::cout << "candidates: " << bitset_util::to_string(candidates) << std::endl;
  }

  return bitset_util::choose_random_on_index(candidates);
}

}  // namespace c4
