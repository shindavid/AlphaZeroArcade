#include <common/MctsPlayer.hpp>

#include <util/BitSet.hpp>
#include <util/Exception.hpp>
#include <util/PrintUtil.hpp>
#include <util/Random.hpp>
#include <util/RepoUtil.hpp>
#include <util/TorchUtil.hpp>

namespace common {

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline MctsPlayer<GameState_, Tensorizor_>::MctsPlayer(const Params& params, Mcts* mcts)
: base_t("CPU")
, params_(params)
, mcts_(mcts ? mcts : new Mcts())
, inv_temperature_(params.temperature ? (1.0 / params.temperature) : 0)
, owns_mcts_(mcts==nullptr)
{
  sim_params_.tree_size_limit = params.num_mcts_iters;
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
    const player_array_t& players, player_index_t seat_assignment)
{
  my_index_ = seat_assignment;
  tensorizor_.clear();
  if (owns_mcts_) {
    mcts_->start();
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline void MctsPlayer<GameState_, Tensorizor_>::receive_state_change(
    player_index_t player, const GameState& state, action_index_t action, const GameOutcome& outcome)
{
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
  mcts_results_ = mcts_->sim(tensorizor_, state, sim_params_);
  GlobalPolicyProbDistr policy = mcts_results_->counts.template cast<float>();
  if (inv_temperature_) {
    policy = policy.pow(inv_temperature_);
  } else {
    policy = (policy == policy.maxCoeff()).template cast<float>();
  }

  ValueProbDistr value = mcts_results_->win_rates;
  if (verbose_info_) {
    policy /= policy.sum();
    verbose_info_->mcts_value = value;
    GameStateTypes::global_to_local(policy, valid_actions, verbose_info_->mcts_policy);
    verbose_info_->mcts_results = *mcts_results_;
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
inline void MctsPlayer<GameState_, Tensorizor_>::get_cache_stats(
    int& hits, int& misses, int& size, float& hash_balance_factor) const
{
  mcts_->get_cache_stats(hits, misses, size, hash_balance_factor);
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
