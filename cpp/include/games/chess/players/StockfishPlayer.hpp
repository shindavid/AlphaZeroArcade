#pragma once

#include "games/chess/players/UciPlayer.hpp"

namespace a0achess {

class StockfishPlayer : public UciPlayer {
 public:
  static inline const ProcParams kDefaultProcParams = {
    .cmd = "extra_deps/stockfish/stockfish-ubuntu-x86-64-avx2",
    .extra_args = ""
  };

  static Params default_params() {
    return Params{.num_procs = 8, .movetime = -1, .depth = 20, .nodes = -1};
  }

  StockfishPlayer(UciPool* pool, const Params& params,
                  const ProcParams& proc_params = kDefaultProcParams)
      : UciPlayer(pool, params, proc_params) {}
};

}  // namespace a0achess
