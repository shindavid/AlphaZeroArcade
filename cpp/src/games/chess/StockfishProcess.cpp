#include "games/chess/StockfishProcess.hpp"

#include "util/Exceptions.hpp"

#include <chess-library/include/chess.hpp>

#include <string>

namespace a0achess {

StockfishProcess::StockfishProcess() {
  auto dir = boost::filesystem::path("extra_deps/stockfish");
  auto relative_bin = boost::filesystem::path("bin/stockfish-ubuntu-x86-64-avx2");
  auto stockfish_bin = dir / relative_bin;

  if (!boost::filesystem::is_regular_file(stockfish_bin)) {
    throw util::CleanException("File does not exist: {}", stockfish_bin.c_str());
  }

  process_ = new boost::process::child(stockfish_bin.string(), boost::process::std_out > out_,
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

StockfishProcess::~StockfishProcess() {
  if (process_) {
    process_->terminate();
    delete process_;
    process_ = nullptr;
  }
}

Move StockfishProcess::query(int depth, const State& state, const MoveSet& valid_moves) {
  in_ << "position fen " << state.getFen() << std::endl;
  in_ << "go depth " << depth << std::endl;

  std::string line;
  while (std::getline(out_, line)) {
    if (line.starts_with("bestmove")) {
      // "bestmove e2e4 ponder d7d5" or "bestmove e2e4"
      auto uci_str = line.substr(9, line.find(' ', 9) - 9);
      return chess::uci::uciToMove(state, uci_str);
    }
  }

  throw util::CleanException("Stockfish process closed unexpectedly");
}

}  // namespace a0achess
