#include <games/connect4/PerfectOracle.hpp>

namespace bp = boost::process;

namespace c4 {

namespace detail {

std::string make_cmd() {
  auto extra_dir = boost::filesystem::path("extra_deps/connect4");
  auto c4_solver_bin = extra_dir / "c4solver";
  auto c4_solver_book = extra_dir / "7x6.book";

  for (const auto& path : {c4_solver_bin, c4_solver_book}) {
    if (!boost::filesystem::is_regular_file(path)) {
      throw util::CleanException("File does not exist: %s", path.c_str());
    }
  }

  return std::format("{} -b {} -a", c4_solver_bin.string(), c4_solver_book.string());
}

}  // namespace detail

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

PerfectOracle::QueryResult PerfectOracle::query(const MoveHistory& history) {
  std::string s;
  {
    std::unique_lock lock(mutex_);
    history.write(in_pipe_);
    std::getline(out_pipe_, s);
  }

  // TODO: do a more specialized parse that avoid dynamic allocation
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

PerfectOracle::PerfectOracle()
    : out_pipe_(),
      in_pipe_(),
      child_(detail::make_cmd(), bp::std_out > out_pipe_,
             bp::std_in < in_pipe_, bp::std_err > bp::null) {}

PerfectOracle::~PerfectOracle() {
  child_.terminate();
}

void PerfectOraclePool::set_capacity(int capacity) {
  util::release_assert(count_ == 0, "Cannot change capacity after oracles are created");
  capacity_ = capacity;
}

}  // namespace c4
