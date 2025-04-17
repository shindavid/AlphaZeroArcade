#include <games/connect4/PerfectOracle.hpp>

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

inline void PerfectOracle::MoveHistory::append(int move) {
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

inline PerfectOracle::PerfectOracle() {
  auto extra_dir = boost::filesystem::path("extra_deps/connect4");
  auto c4_solver_bin = extra_dir / "c4solver";
  auto c4_solver_book = extra_dir / "7x6.book";

  for (const auto& path : {c4_solver_bin, c4_solver_book}) {
    if (!boost::filesystem::is_regular_file(path)) {
      throw util::CleanException("File does not exist: %s", path.c_str());
    }
  }

  namespace bp = boost::process;
  std::string c4_cmd =
    util::create_string("%s -b %s -a", c4_solver_bin.c_str(), c4_solver_book.c_str());
  proc_ = new bp::child(c4_cmd, bp::std_out > out_, bp::std_err > bp::null, bp::std_in < in_);
}

inline PerfectOracle* PerfectOracle::get_instance() {
  std::unique_lock lock(static_mutex_);
  if (oracles_.empty() || oracles_.back()->client_count_ >= kNumClientsPerOracle) {
    oracles_.push_back(new PerfectOracle());
  }
  auto oracle = oracles_.back();
  oracle->client_count_++;
  return oracle;
}

inline PerfectOracle::~PerfectOracle() { delete proc_; }

}  // namespace c4
