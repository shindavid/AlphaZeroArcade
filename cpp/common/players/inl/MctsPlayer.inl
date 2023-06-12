#include <common/players/MctsPlayer.hpp>

#include <unistd.h>

#include <util/BitSet.hpp>
#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>
#include <util/Exception.hpp>
#include <util/Math.hpp>
#include <util/ParamDumper.hpp>
#include <util/Random.hpp>
#include <util/RepoUtil.hpp>
#include <util/ScreenUtil.hpp>
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
    num_raw_policy_starting_moves_distr_mean_str = "0";
  } else if (type == kTraining) {
    num_fast_iters = 100;
    num_full_iters = 600;
    full_pct = 0.25;
    move_temperature_str = "0.8->0.2:2*sqrt(b)";
    num_raw_policy_starting_moves_distr_mean_str = "0.07*m";
  } else {
    throw util::Exception("Unknown type: %d", (int)type);
  }
}


template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void MctsPlayer<GameState_, Tensorizor_>::Params::dump() const {
  if (full_pct == 0) {
    util::ParamDumper::add("MctsPlayer num iters", "%d", num_fast_iters);
  } else {
    util::ParamDumper::add("MctsPlayer num fast iters", "%d", num_fast_iters);
    util::ParamDumper::add("MctsPlayer num full iters", "%d", num_full_iters);
    util::ParamDumper::add("MctsPlayer num fast iters", "%.8g", full_pct);
  }
  util::ParamDumper::add("MctsPlayer move temperature", "%s", move_temperature_str.c_str());
  util::ParamDumper::add("MctsPlayer num raw policy starting moves distr mean", "%s",
                         num_raw_policy_starting_moves_distr_mean_str.c_str());
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
      .template add_option<"move-temp", 't'>(
          po::value<std::string>(&move_temperature_str)->default_value(move_temperature_str),
          "temperature for move selection")
      .template add_option<"num-raw-policy-starting-moves-distr-mean", 'r'>(
          po::value<std::string>(&num_raw_policy_starting_moves_distr_mean_str)->default_value(
              num_raw_policy_starting_moves_distr_mean_str),
              "exponential distribution mean for num raw policy starting moves")
      .template add_option<"verbose", 'v'>(po::bool_switch(&verbose)->default_value(verbose),
          "mcts player verbose mode")
      ;
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline MctsPlayer<GameState_, Tensorizor_>::MctsPlayer(const Params& params, Mcts* mcts)
: params_(params)
, mcts_(mcts)
, search_params_{
        {params.num_fast_iters, true},  // kFast
        {params.num_full_iters},  // kFull
        {1, true}  // kRawPolicy
  }
, move_temperature_(math::ExponentialDecay::parse(params.move_temperature_str, GameStateTypes::get_var_bindings()))
, num_raw_policy_starting_moves_distr_mean_(math::parse_expression(
      params.num_raw_policy_starting_moves_distr_mean_str.c_str(), GameStateTypes::get_var_bindings()))
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
  if (owns_shared_data_) delete shared_data_;
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline void MctsPlayer<GameState_, Tensorizor_>::start_game()
{
  move_count_ = 0;
  util::clean_assert(shared_data_, "shared_data_ not set");
  if (owns_shared_data_) {
    float lambda = 1.0 / num_raw_policy_starting_moves_distr_mean_;
    shared_data_->num_raw_policy_starting_moves = std::floor(util::Random::exponential(lambda));
  }
  move_temperature_.reset();
  tensorizor_.clear();
  if (owns_mcts_) {
    mcts_->start();
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline void MctsPlayer<GameState_, Tensorizor_>::receive_state_change(
    seat_index_t seat, const GameState& state, action_index_t action)
{
  move_count_++;
  move_temperature_.step();
  tensorizor_.receive_state_change(state, action);
  if (owns_mcts_) {
    mcts_->receive_state_change(seat, state, action);
  }
  if (base_t::get_my_seat() == seat && params_.verbose) {
    if (facing_human_tui_player_) {
      util::ScreenClearer::clear_once();
    }
    verbose_dump();
    if (!facing_human_tui_player_) {
      state.dump(action, &this->get_player_names());
    }
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline action_index_t MctsPlayer<GameState_, Tensorizor_>::get_action(
    const GameState& state, const ActionMask& valid_actions)
{
  SearchMode search_mode = choose_search_mode();
  const MctsResults* mcts_results = mcts_search(state, search_mode);
  return get_action_helper(search_mode, mcts_results, valid_actions);
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline void MctsPlayer<GameState_, Tensorizor_>::get_cache_stats(
    int& hits, int& misses, int& size, float& hash_balance_factor) const
{
  mcts_->get_cache_stats(hits, misses, size, hash_balance_factor);
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline MctsPlayer<GameState_, Tensorizor_>::SharedData* MctsPlayer<GameState_, Tensorizor_>::init_shared_data() {
  owns_shared_data_ = true;
  shared_data_ = new SharedData();
  return shared_data_;
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline void MctsPlayer<GameState_, Tensorizor_>::set_shared_data(SharedData* shared_data) {
  shared_data_ = shared_data;
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline const typename MctsPlayer<GameState_, Tensorizor_>::MctsResults*
MctsPlayer<GameState_, Tensorizor_>::mcts_search(const GameState& state, SearchMode search_mode) const {
  return mcts_->search(tensorizor_, state, search_params_[search_mode]);
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline typename MctsPlayer<GameState_, Tensorizor_>::SearchMode
MctsPlayer<GameState_, Tensorizor_>::choose_search_mode() const {
  bool use_raw_policy = move_count_ < shared_data_->num_raw_policy_starting_moves;
  return use_raw_policy ? kRawPolicy : get_random_search_mode();
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline action_index_t MctsPlayer<GameState_, Tensorizor_>::get_action_helper(
    SearchMode search_mode, const MctsResults* mcts_results, const ActionMask& valid_actions) const
{
  PolicyTensor policy_tensor;
  ValueArray value;
  if (search_mode == kRawPolicy) {
    GameStateTypes::local_to_global(mcts_results->policy_prior, valid_actions, policy_tensor);
    value = mcts_results->value_prior;
  } else {
    policy_tensor = mcts_results->counts;
    value = mcts_results->win_rates;
  }

  PolicyArray& policy = eigen_util::reinterpret_as_array(policy_tensor);
  if (search_mode != kRawPolicy) {
    float temp = move_temperature_.value();
    if (temp != 0) {
      policy = policy.pow(1.0 / temp);
    } else {
      policy = (policy == policy.maxCoeff()).template cast<torch_util::dtype>();
    }
  }

  if (policy.sum() == 0) {
    // This happens if eliminations are enabled and if MCTS proves that the position is losing.
    // In this case we just choose a random valid action.
    for (action_index_t action : bitset_util::on_indices(valid_actions)) {
      policy[action] = 1;
    }
  }

  if (verbose_info_) {
    policy /= policy.sum();
    verbose_info_->mcts_value = value;
    GameStateTypes::global_to_local(policy_tensor, valid_actions, verbose_info_->mcts_policy);
    verbose_info_->mcts_results = *mcts_results;
    verbose_info_->initialized = true;
  }
  action_index_t action = util::Random::weighted_sample(policy.begin(), policy.end());
  assert(valid_actions[action]);
  return action;
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
MctsPlayer<GameState_, Tensorizor_>::SearchMode
MctsPlayer<GameState_, Tensorizor_>::get_random_search_mode() const {
  float r = util::Random::uniform_real<float>(0.0f, 1.0f);
  return r < params_.full_pct ? kFull : kFast;
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline void MctsPlayer<GameState_, Tensorizor_>::verbose_dump() const {
  if (!verbose_info_->initialized) return;

  const auto& mcts_value = verbose_info_->mcts_value;
  const auto& mcts_policy = verbose_info_->mcts_policy;
  const auto& mcts_results = verbose_info_->mcts_results;

  printf("CPU pos eval:\n");
  GameState::dump_mcts_output(mcts_value, mcts_policy, mcts_results);
  std::cout << std::endl;
}

}  // namespace common
