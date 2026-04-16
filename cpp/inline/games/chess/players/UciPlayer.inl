#include "games/chess/players/UciPlayer.hpp"

#include "games/chess/UciProcess.hpp"

#include <boost/type_traits/add_lvalue_reference.hpp>

namespace a0achess {

inline auto UciPlayer::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("UciPlayer options");
  return desc
    .template add_option<"movetime", 'm'>(po::value<int>(&movetime)->default_value(movetime),
                                          "Move time in milliseconds")
    .template add_option<"depth", 'd'>(po::value<int>(&depth)->default_value(depth), "Search depth")
    .template add_option<"nodes", 'n'>(po::value<int>(&nodes)->default_value(nodes),
                                       "Number of nodes to search")
    .template add_option<"num-procs", 'p'>(
      po::value<int>(&num_procs)->default_value(num_procs),
      "number of UCI processes to use (defaults to number of game threads)");
  ;
}

inline UciPlayer::ActionResponse UciPlayer::get_action_response(const ActionRequest& request) {
  if (request.aux) {
    return Move(request.aux);
  }

  UciProcess* proc = pool_->get_oracle(request.notification_unit, proc_params_);
  if (!proc) {
    return ActionResponse::yield();
  }

  std::string uci_str = proc->query(move_str_, go_cmd_);
  Move move = Move::from_str(request.info_set, uci_str);
  pool_->release_oracle(proc);
  ActionResponse response(move);
  response.set_aux(move.move());
  return response;
}

inline void UciPlayer::receive_state_change(const StateChangeUpdate& update) {
  if (!update.is_jump()) {
    move_str_ += " ";
    move_str_ += update.move()->to_str();
  } else {
    move_str_.clear();
    auto state_it = update.state_it();

    std::vector<const Move*> moves;
    moves.reserve(state_it->step);

    while (state_it->move_from_parent) {
      moves.push_back(state_it->move_from_parent);
      ++state_it;
    }
    for (auto it = moves.rbegin(); it != moves.rend(); ++it) {
      move_str_ += ' ';
      move_str_ += (*it)->to_str();
    }
  }
}

inline std::string UciPlayer::Params::build_go_command() const {
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
