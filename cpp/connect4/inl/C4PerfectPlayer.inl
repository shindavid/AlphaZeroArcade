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
  }

  QueryResult result{best_moves, good_moves, best_score};
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

inline void PerfectGrader::stats_t::update(bool correct) {
  correct_count += correct;
  total_count++;
}

inline PerfectGrader::stats_t& PerfectGrader::stats_t::operator+=(const stats_t& rhs) {
  correct_count += rhs.correct_count;
  total_count += rhs.total_count;
  return *this;
}

inline void PerfectGrader::Listener::on_game_start(common::game_id_t) {
  move_history_.reset();
  move_number_ = 0;
}

inline void PerfectGrader::Listener::on_game_end() {
  move_number_t total_moves = move_number_;
  for (auto it : tmp_stats_map_) {
    int player = std::get<0>(it.first);
    int move = std::get<1>(it.first);

    for (common::player_index_t p : {player, -1}) {
      for (move_number_t m : {move, 0, move - total_moves - 1}) {
        grader_.stats_map()[std::make_tuple(p, m)] += it.second;
      }
    }
  }
  tmp_stats_map_.clear();
}

inline void PerfectGrader::Listener::on_move(common::player_index_t player, common::action_index_t action) {
  ++move_number_;
  PerfectOracle::QueryResult result = grader_.oracle().query(move_history_);

  if (result.score >= 0) {
    bool correct = result.good_moves[action];
    tmp_stats_map_[std::make_tuple(player, move_number_)].update(correct);
  }

  move_history_.append(action);
}

inline void PerfectGrader::dump() const {
  for (auto it : stats_map_) {
    auto player = std::get<0>(it.first);
    auto move = std::get<1>(it.first);
    printf("PerfectGrader player=%d move=%d correct=%d total=%d\n",
           player, move, it.second.correct_count, it.second.total_count);
  }
}

}  // namespace c4
