#include "games/connect4/PerfectOracle.hpp"

namespace bp = boost::process;

namespace c4 {

namespace detail {

std::string make_cmd() {
  auto extra_dir = boost::filesystem::path("extra_deps/connect4");
  auto c4_solver_bin = extra_dir / "c4solver";
  auto c4_solver_book = extra_dir / "7x6.book";

  for (const auto& path : {c4_solver_bin, c4_solver_book}) {
    if (!boost::filesystem::is_regular_file(path)) {
      throw util::CleanException("File does not exist: {}", path.c_str());
    }
  }

  return std::format("{} -b {} -a", c4_solver_bin.string(), c4_solver_book.string());
}

}  // namespace detail

std::string PerfectOracle::QueryResult::get_overlay() const {
  char chars[2 * kNumColumns + 1];
  char array[3] = {'-', '0', '+'};

  char* c = chars;
  for (int i = 0; i < kNumColumns; ++i) {
    int score = scores[i];
    *(c++) = ' ';
    int k = (score < 0) ? 0 : (score > 0) ? 2 : 1;
    *(c++) = array[k];
  }
  *c = '\0';
  return std::string(chars);
}

PerfectOracle::QueryResult PerfectOracle::query(const MoveHistory& history) {
  {
    mit::unique_lock lock(mutex_);
    history.write(in_);
    std::getline(out_, output_str_);
  }

  // TODO: do a more specialized parse that avoid dynamic allocation
  int n_tokens = util::split(tokens_, output_str_);

  QueryResult result;
  for (int j = 0; j < kNumColumns; ++j) {
    int raw_score = std::stoi(tokens_[n_tokens - kNumColumns + j]);
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

PerfectOracle::PerfectOracle()
    : out_(),
      in_(),
      child_(detail::make_cmd(), bp::std_out > out_, bp::std_in<in_, bp::std_err> bp::null) {}

PerfectOracle::~PerfectOracle() { child_.terminate(); }

}  // namespace c4
