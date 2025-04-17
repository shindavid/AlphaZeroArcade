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

inline void PerfectOracle::MoveHistory::write(boost::process::opstream& in) const {
  *char_pointer_ = '\n';
  in.write(chars_, char_pointer_ - chars_ + 1);
  in.flush();
}

inline void PerfectOracle::async_query(const MoveHistory& history) {
  history.write(in_pipe_);
  history_length_ = history.length();
}

inline PerfectOracle* PerfectOraclePool::get_oracle() {
  std::unique_lock lock(mutex_);
  if (!free_oracles_.empty()) {
    PerfectOracle* oracle = free_oracles_.back();
    free_oracles_.pop_back();
    return oracle;
  }
  if (count_ < capacity_) {
    count_++;
    PerfectOracle* oracle = new PerfectOracle();
    return oracle;
  }
  return nullptr;
}

inline void PerfectOraclePool::release_oracle(PerfectOracle* oracle) {
  std::unique_lock lock(mutex_);
  free_oracles_.push_back(oracle);
}

}  // namespace c4
