#include "games/chess/UciProcess.hpp"

#include "util/Exceptions.hpp"

#include <chess-library/include/chess.hpp>

#include <filesystem>
#include <string>

namespace a0achess {

UciProcess::UciProcess(const Params& params) {
  if (!std::filesystem::is_regular_file(params.cmd)) {
    throw util::CleanException("File does not exist: {}", params.cmd);
  }

  auto cmd = std::format("{} {}", params.cmd, params.extra_args);

  process_ =
    new boost::process::child(cmd, boost::process::std_out > out_, boost::process::std_in < in_);

  in_ << params.uci_settings << std::endl;

  std::string line;
  in_ << "isready" << std::endl;
  while (std::getline(out_, line)) {
    if (line.starts_with("readyok")) break;
  }
}

UciProcess::~UciProcess() {
  if (process_) {
    process_->terminate();
    delete process_;
    process_ = nullptr;
  }
}

std::string UciProcess::query(const std::string& fen_move_str, const std::string& go_cmd) {
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

  throw util::CleanException("UCI process closed unexpectedly");
}

std::string UciProcess::parse_bestmove_line(const std::string& line) {
  size_t space_pos = line.find(' ', 9);
  return line.substr(9, space_pos == std::string::npos ? std::string::npos : space_pos - 9);
}

}  // namespace a0achess
