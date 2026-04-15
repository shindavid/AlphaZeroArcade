#include "games/chess/LcZeroProcess.hpp"

#include "util/Exceptions.hpp"

#include <chess-library/include/chess.hpp>

#include <string>

namespace a0achess {

LcZeroProcess::LcZeroProcess() {

  auto dir = boost::filesystem::path("extra_deps/lc0");
  auto relative_bin = boost::filesystem::path("lc0");
  auto lc0_bin = dir / relative_bin;
  auto weights_file = boost::filesystem::path("BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz");

  if (!boost::filesystem::is_regular_file(lc0_bin)) {
    throw util::CleanException("File does not exist: {}", lc0_bin.c_str());
  }

  auto cmd = std::format("{} --weights={}", lc0_bin.string(), (dir / weights_file).string());

  process_ = new boost::process::child(cmd, boost::process::std_out > out_,
                                       boost::process::std_in < in_);

  constexpr const char* kSyzygyPath = "/workspace/syzygy";
  if (boost::filesystem::is_directory(kSyzygyPath)) {
    in_ << "setoption name SyzygyPath value " << kSyzygyPath << std::endl;
  }

  std::string line;
  in_ << "isready" << std::endl;
  while (std::getline(out_, line)) {
    if (line.starts_with("readyok")) break;
  }
}

LcZeroProcess::~LcZeroProcess() {
  if (process_) {
    process_->terminate();
    delete process_;
    process_ = nullptr;
  }
}

std::string LcZeroProcess::query(const std::string& fen_move_str, const std::string& go_cmd) {
  in_ << "position startpos moves" << fen_move_str << std::endl;
  in_ << go_cmd << std::endl;

  std::string line;
  while (std::getline(out_, line)) {
    if (line.starts_with("bestmove")) {
      // "bestmove e2e4 ponder d7d5" or "bestmove e2e4"
      auto uci_str = parse_bestmove_line(line);
      return uci_str;
    }
  }

  throw util::CleanException("LcZero process closed unexpectedly");
}

std::string LcZeroProcess::parse_bestmove_line(const std::string& line) {
  // Using the safer parsing logic to prevent integer underflow
  size_t space_pos = line.find(' ', 9);
  return line.substr(9, space_pos == std::string::npos ? std::string::npos : space_pos - 9);
}

}  // namespace a0achess
