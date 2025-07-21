#include "games/othello/players/EdaxPlayer.hpp"
#include "util/BoostUtil.hpp"

namespace othello {

inline auto EdaxPlayer::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("othello::EdaxPlayer options");
  return desc
    .template add_option<"depth", 'd'>(po::value<int>(&depth)->default_value(depth), "Search depth")
    .template add_option<"deterministic", 'D'>(
      po::bool_switch(&deterministic)->default_value(deterministic),
      "edax player deterministic mode")
    .template add_option<"verbose", 'v'>(po::bool_switch(&verbose)->default_value(verbose),
                                         "edax player verbose mode")
    .template add_option<"num-oracle-procs", 'n'>(
      po::value<int>(&num_oracle_procs)->default_value(num_oracle_procs),
      "number of oracle processes to use (defaults to number of game threads)");
}

inline EdaxPlayer::EdaxPlayer(OraclePool* oracle_pool, const Params& params)
    : oracle_pool_(oracle_pool), params_(params) {
  CLEAN_ASSERT(params_.depth >= 0 && params_.depth <= 21, "edax depth must be in [0, 21]");
}

inline EdaxPlayer::ActionResponse EdaxPlayer::get_action_response(const ActionRequest& request) {
  const auto& state = request.state;
  const auto& valid_actions = request.valid_actions;
  int num_valid_actions = valid_actions.count();

  if (num_valid_actions == 1) {  // only 1 possible move, no need to incur edax/IO overhead
    core::action_t action = bitset_util::get_nth_on_index(valid_actions, 0);
    return action;
  }

  EdaxOracle* oracle =
    oracle_pool_->get_oracle(request.notification_unit, params_.verbose, params_.deterministic);
  if (!oracle) {
    return ActionResponse::yield();
  }

  core::action_t action = oracle->query(params_.depth, state, request.valid_actions);
  oracle_pool_->release_oracle(oracle);
  return action;
}

}  // namespace othello
