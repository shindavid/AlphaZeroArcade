#include "games/chess/players/LcZeroPlayer.hpp"

namespace a0achess {

inline auto LcZeroPlayer::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("LcZeroPlayer options");
  return desc
    .template add_option<"movetime", 'm'>(po::value<int>(&movetime)->default_value(movetime), "Move time in milliseconds")
    .template add_option<"nodes", 'n'>(po::value<int>(&nodes)->default_value(nodes), "Number of nodes to search")
    .template add_option<"num-procs", 'p'>(
      po::value<int>(&num_procs)->default_value(num_procs),
      "number of lc0 processes to use (defaults to number of game threads)");
  ;
}

inline LcZeroPlayer::ActionResponse LcZeroPlayer::get_action_response(
  const ActionRequest& request) {
  if (request.aux) {
    return Move(request.aux);
  }

  LcZeroProcess* lc0_proc = lc0_pool_->get_oracle(request.notification_unit);
  if (!lc0_proc) {
    return ActionResponse::yield();
  }

  std::string uci_str = lc0_proc->query(get_fen_move(), params_.build_go_command());
  Move move = Move::from_str(request.state, uci_str);
  lc0_pool_->release_oracle(lc0_proc);
  ActionResponse response(move);
  response.set_aux(move.move());
  return response;
}

inline std::string LcZeroPlayer::get_fen_move() const {
  std::string move_strs;
  for (auto v : move_value_history_) {
    Move m = Move(v);
    move_strs += " " + m.to_str();
  }
  return move_strs;
}

inline std::string LcZeroPlayer::Params::build_go_command() const {
  std::ostringstream oss;
  oss << "go";

  for (const auto& [name, ptr] : go_options) {
    if (this->*ptr > 0) {
      oss << " " << name << " " << this->*ptr;
    }
  }
  return oss.str();
}

}  // namespace a0achess
