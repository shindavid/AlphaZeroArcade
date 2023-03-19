#include <connect4/C4PerfectPlayer.hpp>

#include <util/BitSet.hpp>
#include <util/BoostUtil.hpp>
#include <util/Config.hpp>
#include <util/Exception.hpp>
#include <util/RepoUtil.hpp>
#include <util/StringUtil.hpp>

namespace c4 {

inline auto PerfectPlayParams::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  std::string default_c4_solver_dir = util::Config::instance()->get("c4.solver_dir", "");

  auto c4_solver_dir_value = po::value<std::string>(&c4_solver_dir);
  if (!default_c4_solver_dir.empty()) {
    c4_solver_dir_value = c4_solver_dir_value->default_value(default_c4_solver_dir);
  }

  po2::options_description desc("C4PerfectPlayer options");
  return desc
      .template add_option<"c4-solver-dir", 'c'>(c4_solver_dir_value, "base dir containing c4solver bin+book")
      .template add_option<"leisurely-mode", 'l'>(po::bool_switch(&leisurely_mode)->default_value(leisurely_mode),
          "exhibit no preference among winning moves as perfect player")
  ;
}

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
    if (score<0 || !good_moves[i]) {
      chars[i] = drawing_moves[i] ? '0' : ' ';
    } else {
      chars[i] = drawing_moves[i] ? '0' : '+';
    }
  }
  return util::create_string(" %c %c %c %c %c %c %c",
                             chars[0], chars[1], chars[2], chars[3],
                             chars[4], chars[5], chars[6]);
}

inline PerfectOracle::PerfectOracle(const PerfectPlayParams& params) {
  if (params.c4_solver_dir.empty()) {
    throw util::Exception("c4 solver dir not specified! Please add 'c4.solver_dir' entry in $REPO_ROOT/%s",
                          util::Config::kFilename);
  }
  boost::filesystem::path c4_solver_dir(params.c4_solver_dir);
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
  int move_scores[kNumColumns];
  for (int j = 0; j < kNumColumns; ++j) {
    int score = std::stoi(tokens[tokens.size() - kNumColumns + j]);
    move_scores[j] = score;
  }
  int best_score = *std::max_element(move_scores, move_scores + kNumColumns);

  ActionMask good_moves;
  ActionMask best_moves;
  ActionMask drawing_moves;

  int good_bound;
  int best_bound = best_score;
  if (best_score > 0) {
    good_bound = 1;
  } else if (best_score == 0) {
    good_bound = 0;
  } else {
    good_bound = -999;  // -1000 indicates illegal move
  }

  for (int j = 0; j < c4::kNumColumns; ++j) {
    best_moves[j] = move_scores[j] >= best_bound;
    good_moves[j] = move_scores[j] >= good_bound;
    drawing_moves[j] = move_scores[j] == 0;
  }

  int converted_score = best_score;
  if (best_score > 0) {
    converted_score = 22 - history.length() / 2 - best_score;
    if (converted_score <= 0) {
      throw util::Exception("Bad score conversion (score=%d, history=%s(%d), converted_score=%d)",
                            best_score, history.to_string().c_str(), history.length(), converted_score);
    }
  } else if (best_score < 0) {
    converted_score = -22 + (history.length() + 1) / 2 - best_score;
    if (converted_score >= 0) {
      throw util::Exception("Bad score conversion (score=%d, history=%s(%d), converted_score=%d)",
                            best_score, history.to_string().c_str(), history.length(), converted_score);
    }
  }

  QueryResult result{best_moves, good_moves, drawing_moves, converted_score};
  return result;
}

inline PerfectPlayer::PerfectPlayer(const PerfectPlayParams& params)
  : base_t("Perfect")
  , oracle_(params)
  , leisurely_mode_(params.leisurely_mode) {}

inline void PerfectPlayer::start_game(
    common::game_id_t game_id, const player_array_t& players, common::player_index_t seat_assignment)
{
  move_history_.reset();
}

inline void PerfectPlayer::receive_state_change(
    common::player_index_t, const GameState&, common::action_index_t action, const GameOutcome&)
{
  move_history_.append(action);
}

inline common::action_index_t PerfectPlayer::get_action(const GameState&, const ActionMask&) {
  auto result = oracle_.query(move_history_);
  if (leisurely_mode_ && result.score > 0) {
    return bitset_util::choose_random_on_index(result.good_moves);
  } else {
    return bitset_util::choose_random_on_index(result.best_moves);
  }
}

}  // namespace c4
