#pragma once

#include "games/chess/Game.hpp"

#include <boost/process.hpp>

namespace a0achess {

class StockfishProcess {
 public:
  using State = Game::State;

  StockfishProcess();
  ~StockfishProcess();

  std::string query(int depth, const std::string& fen_move_str);

 private:
  boost::process::child* process_;
  boost::process::ipstream out_;
  boost::process::opstream in_;
};

}  // namespace a0achess
