#include "games/chess/players/StockfishPlayer.hpp"

namespace a0achess {

inline auto StockfishPlayer::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("StockfishPlayer options");
  return desc
    .template add_option<"depth", 'd'>(po::value<int>(&depth)->default_value(depth), "Search depth")
    .template add_option<"num-oracle-procs", 'n'>(
      po::value<int>(&num_stockfish_procs)->default_value(num_stockfish_procs),
      "number of stockfish processes to use (defaults to number of game threads)");
  ;
}

inline StockfishPlayer::ActionResponse StockfishPlayer::get_action_response(
  const ActionRequest& request) {
  if (request.aux) {
    return Move(request.aux);
  }

  const auto& state = request.state;

  StockfishProcess* stockfish_proc = stockfish_pool_->get_oracle(request.notification_unit);
  if (!stockfish_proc) {
    return ActionResponse::yield();
  }

  Move move = stockfish_proc->query(params_.depth, state, request.valid_moves);
  stockfish_pool_->release_oracle(stockfish_proc);
  ActionResponse response(move);
  response.set_aux(move.move());
  return response;
}

}  // namespace a0achess
