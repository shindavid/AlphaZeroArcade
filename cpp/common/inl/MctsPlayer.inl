#include <common/MctsPlayer.hpp>

#include <util/BitSet.hpp>
#include <util/BoostUtil.hpp>
#include <util/Exception.hpp>
#include <util/Math.hpp>
#include <util/PrintUtil.hpp>
#include <util/Random.hpp>
#include <util/RepoUtil.hpp>
#include <util/StringUtil.hpp>
#include <util/TorchUtil.hpp>

namespace common {

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
MctsPlayer<GameState_, Tensorizor_>::Params::Params(DefaultParamsType type)
{
  if (type == kCompetitive) {
    num_fast_iters = 1600;
    num_full_iters = 0;
    full_pct = 0.0;
    move_temperature_str = "0.5->0.2:2*sqrt(b)";
  } else if (type == kTraining) {
    num_fast_iters = 100;
    num_full_iters = 600;
    full_pct = 0.25;
    move_temperature_str = "0.8->0.2:2*sqrt(b)";
  } else {
    throw util::Exception("Unknown type: %d", (int)type);
  }
}


template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void MctsPlayer<GameState_, Tensorizor_>::Params::dump() const {
  if (full_pct == 0) {
    PARAM_DUMP("MctsPlayer num iters", "%d", num_fast_iters);
  } else {
    PARAM_DUMP("MctsPlayer num fast iters", "%d", num_fast_iters);
    PARAM_DUMP("MctsPlayer num full iters", "%d", num_full_iters);
    PARAM_DUMP("MctsPlayer num fast iters", "%.8g", full_pct);
    PARAM_DUMP("MctsPlayer move temperature", "%s", move_temperature_str.c_str());
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
auto MctsPlayer<GameState_, Tensorizor_>::Params::make_options_description()
{
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("MctsPlayer options");

  return desc
      .template add_option<"num-fast-iters", 'i'>(po::value<int>(&num_fast_iters)->default_value(num_fast_iters),
          "num mcts iterations to do per fast move")
      .template add_option<"num-full-iters", 'I'>(po::value<int>(&num_full_iters)->default_value(num_full_iters),
          "num mcts iterations to do per full move")
      .template add_option<"full-pct", 'f'>(po2::float_value("%.2f", &full_pct, full_pct),
          "pct of moves that should be full")
      .template add_option<"move-temp", 't'>(po::value<std::string>(&move_temperature_str)->default_value(move_temperature_str),
          "temperature for move selection")
      .template add_option<"verbose", 'v'>(po::bool_switch(&verbose)->default_value(verbose),
          "mcts player verbose mode")
      ;
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline MctsPlayer<GameState_, Tensorizor_>::MctsPlayer(const Params& params, Mcts* mcts)
: base_t("MCTS")
, params_(params)
, mcts_(mcts)
, sim_params_{
        {params.num_fast_iters, true},  // kFast
        {params.num_full_iters},  // kFull
        {1, true}  // kRawPolicy
  }
, move_temperature_(math::ExponentialDecay::parse(params.move_temperature_str, GameStateTypes::get_var_bindings()))
, owns_mcts_(mcts==nullptr)
{
  if (params.verbose) {
    verbose_info_ = new VerboseInfo();
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
template<typename... Ts>
MctsPlayer<GameState_, Tensorizor_>::MctsPlayer(const Params& params, Ts&&... mcts_params_args)
: MctsPlayer(params, new Mcts(std::forward<Ts>(mcts_params_args)...))
{
  owns_mcts_ = true;
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline MctsPlayer<GameState_, Tensorizor_>::~MctsPlayer() {
  if (verbose_info_) {
    delete verbose_info_;
  }
  if (owns_mcts_) delete mcts_;
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline void MctsPlayer<GameState_, Tensorizor_>::start_game(
    game_id_t, const player_array_t& players, player_index_t seat_assignment)
{
  my_index_ = seat_assignment;
  move_count_ = 0;

  move_temperature_.reset();
  tensorizor_.clear();
  if (owns_mcts_) {
    mcts_->start();
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline void MctsPlayer<GameState_, Tensorizor_>::receive_state_change(
    player_index_t player, const GameState& state, action_index_t action, const GameOutcome& outcome)
{
  move_count_++;
  move_temperature_.step();
  tensorizor_.receive_state_change(state, action);
  if (owns_mcts_) {
    mcts_->receive_state_change(player, state, action, outcome);
  }
  if (my_index_ == player && params_.verbose) {
    verbose_dump();
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline action_index_t MctsPlayer<GameState_, Tensorizor_>::get_action(
    const GameState& state, const ActionMask& valid_actions)
{
  SimType sim_type = choose_sim_type();
  const MctsResults* mcts_results = mcts_sim(state, sim_type);
  return get_action_helper(sim_type, mcts_results, valid_actions);
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline void MctsPlayer<GameState_, Tensorizor_>::get_cache_stats(
    int& hits, int& misses, int& size, float& hash_balance_factor) const
{
  mcts_->get_cache_stats(hits, misses, size, hash_balance_factor);
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline const typename MctsPlayer<GameState_, Tensorizor_>::MctsResults*
MctsPlayer<GameState_, Tensorizor_>::mcts_sim(const GameState& state, SimType sim_type) const {
  return mcts_->sim(tensorizor_, state, sim_params_[sim_type]);
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline typename MctsPlayer<GameState_, Tensorizor_>::SimType
MctsPlayer<GameState_, Tensorizor_>::choose_sim_type() const {
  bool use_raw_policy = move_count_ < params_.num_raw_policy_starting_moves;
  return use_raw_policy ? kRawPolicy : get_random_sim_type();
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline action_index_t MctsPlayer<GameState_, Tensorizor_>::get_action_helper(
    SimType sim_type, const MctsResults* mcts_results, const ActionMask& valid_actions) const
{
  if (sim_type == kRawPolicy) {
    GlobalPolicyProbDistr raw_policy;
    raw_policy.setConstant(0);
    GameStateTypes::local_to_global(mcts_results->policy_prior, valid_actions, raw_policy);
    return util::Random::weighted_sample(raw_policy.begin(), raw_policy.end());
  }
  GlobalPolicyProbDistr policy = mcts_results->counts.template cast<float>();
  float temp = move_temperature_.value();
  if (temp != 0) {
    policy = policy.pow(1.0 / temp);
  } else {
    policy = (policy == policy.maxCoeff()).template cast<float>();
  }

  ValueProbDistr value = mcts_results->win_rates;
  if (verbose_info_) {
    policy /= policy.sum();
    verbose_info_->mcts_value = value;
    GameStateTypes::global_to_local(policy, valid_actions, verbose_info_->mcts_policy);
    verbose_info_->mcts_results = *mcts_results;
    verbose_info_->initialized = true;
  }
  action_index_t action = util::Random::weighted_sample(policy.begin(), policy.end());
  if (!valid_actions[action]) {
    // This happens rarely, due to MCTS elimination mechanics
    return bitset_util::choose_random_on_index(valid_actions);
  }
  return action;
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
MctsPlayer<GameState_, Tensorizor_>::SimType
MctsPlayer<GameState_, Tensorizor_>::get_random_sim_type() const {
  float r = util::Random::uniform_real<float>(0.0f, 1.0f);
  return r < params_.full_pct ? kFull : kFast;
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline void MctsPlayer<GameState_, Tensorizor_>::verbose_dump() const {
  if (!verbose_info_->initialized) return;

  const auto& mcts_value = verbose_info_->mcts_value;
  const auto& mcts_policy = verbose_info_->mcts_policy;
  const auto& mcts_results = verbose_info_->mcts_results;

  util::xprintf("CPU pos eval:\n");
  GameState::xdump_mcts_output(mcts_value, mcts_policy, mcts_results);
  util::xprintf("\n");
  util::xflush();
}

}  // namespace common
