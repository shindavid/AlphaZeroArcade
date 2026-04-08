#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/OraclePool.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/StockfishProcess.hpp"



#include <boost/asio.hpp>

namespace a0achess {

class StockfishPlayer : public core::AbstractPlayer<Game> {
 public:
  using StockfishPool = core::OraclePool<StockfishProcess>;
  using ActionResponse = core::ActionResponse<Game>;

  struct Params {
    int depth = 10;
    int num_stockfish_procs = 8;
    auto make_options_description();
  };

  StockfishPlayer(StockfishPool* stockfish_pool, const Params& params)
      : stockfish_pool_(stockfish_pool), params_(params) {}

  ActionResponse get_action_response(const ActionRequest& request) override;

 private:
  StockfishPool* const stockfish_pool_;
  Params params_;
};

}  // namespace a0achess

#include "inline/games/chess/players/StockfishPlayer.inl"
