#include <connect4/C4PerfectPlayer.hpp>

#include <util/Config.hpp>
#include <util/Exception.hpp>
#include <util/StringUtil.hpp>

namespace c4 {

inline PerfectOracle::MoveHistory::MoveHistory() : char_pointer_(chars_) {}

inline void PerfectOracle::MoveHistory::reset() {
  char_pointer_ = chars_;
  *char_pointer_ = 0;
}

inline void PerfectOracle::MoveHistory::append(common::action_index_t move) {
  *(char_pointer_++) = char(int('1') + move);  // connect4 program uses 1-indexing
}

inline void PerfectOracle::MoveHistory::write(boost::process::opstream& in) {
  *char_pointer_ = '\n';
  in.write(chars_, char_pointer_ - chars_ + 1);
  in.flush();
}

inline PerfectOracle::PerfectOracle(const boost::filesystem::path &c4_solver_dir) {
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

inline PerfectOracle::QueryResult PerfectOracle::get_best_moves(MoveHistory &history, bool strong) {
  history.write(in_);

  std::string s;
  std::getline(out_, s);
  auto tokens = util::split(s);
  int move_scores[kNumColumns];
  for (int j = 0; j < kNumColumns; ++j) {
    int score = std::stoi(tokens[tokens.size() - kNumColumns + j]);
    move_scores[j] = score;
  }
  int best_score = *std::max_element(move_scores, move_scores + kNumColumns);

  ActionMask best_moves;
  int score_bound = (strong || best_score <= 0) ? best_score : 1;
  for (int j = 0; j < c4::kNumColumns; ++j) {
    best_moves[j] = move_scores[j] >= score_bound;
  }

  QueryResult result{best_moves, best_score};
  return result;
}

inline boost::filesystem::path PerfectOracle::get_default_c4_solver_dir() {
  return util::Config::instance()->get("c4.solver_dir");
}

inline PerfectPlayer::PerfectPlayer(const Params& params)
  : base_t("Perfect")
  , oracle_(params.c4_solver_dir)
  , strong_mode_(params.strong_mode) {}

inline void PerfectPlayer::start_game(const player_array_t& players, common::player_index_t seat_assignment) {
  move_history_.reset();
}

inline void PerfectPlayer::receive_state_change(
    common::player_index_t, const GameState&, common::action_index_t action, const Result&)
{
  move_history_.append(action);
}

inline common::action_index_t PerfectPlayer::get_action(const GameState&, const ActionMask&) {
  ActionMask best_moves = oracle_.get_best_moves(move_history_, strong_mode_).moves;
  return best_moves.choose_random_set_bit();
}

}  // namespace c4
