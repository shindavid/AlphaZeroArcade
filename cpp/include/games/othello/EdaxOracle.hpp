#pragma once

#include "games/othello/Game.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <boost/process.hpp>

#include <string>
#include <vector>

namespace othello {

class EdaxOracle {
 public:
  using State = Game::State;
  using Move = Game::Move;
  using MoveSet = Game::MoveSet;

  EdaxOracle(bool verbose, bool deterministic_mode);
  ~EdaxOracle();

  Move query(int depth, const State& state, const MoveSet& valid_moves);

 private:
  std::vector<std::string> line_buffer_;
  std::vector<std::string> tokens_;

  boost::process::ipstream out_;
  boost::process::opstream in_;
  boost::process::child* child_;

  mutable mit::mutex mutex_;
  const bool verbose_;
  const bool deterministic_mode_;
};

}  // namespace othello
