#pragma once

#include "games/chess/Game.hpp"

#include <boost/process.hpp>

namespace a0achess {

class StockfishProcess {
 public:
  using State = Game::State;

  StockfishProcess();
  ~StockfishProcess();

  Move query(int depth, const State& state, const MoveSet& valid_moves);

 private:
  boost::process::child* process_;
  boost::process::ipstream out_;
  boost::process::opstream in_;
};

}  // namespace a0achess
