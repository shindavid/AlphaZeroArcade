#include <common/NNetPlayer.hpp>

#include <util/Exception.hpp>
#include <util/PrintUtil.hpp>
#include <util/Random.hpp>
#include <util/RepoUtil.hpp>
#include <util/TorchUtil.hpp>

namespace common {

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline NNetPlayer<GameState_, Tensorizor_>::Params::Params()
  : model_filename(util::Repo::root() / "c4_model.pt") {}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline NNetPlayer<GameState_, Tensorizor_>::NNetPlayer(const Params& params)
  : base_t("CPU")
  , params_(params)
  , net_(params.model_filename)
  , policy_(std::array<int64_t, 2>{1, GameState::kNumGlobalActions})
  , value_(std::array<int64_t, 2>{1, GameState::kNumPlayers})
  , mcts_(net_, 1, 0, 4096)
  , inv_temperature_(params.temperature ? (1.0 / params.temperature) : 0)
{
  torch_input_gpu_ = input_.asTorch().clone().to(torch::kCUDA);
  input_vec_.push_back(torch_input_gpu_);

  if (params.verbose) {
    verbose_info_ = new VerboseInfo();
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline NNetPlayer<GameState_, Tensorizor_>::~NNetPlayer() {
  if (verbose_info_) {
    delete verbose_info_;
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline void NNetPlayer<GameState_, Tensorizor_>::start_game(
    const player_array_t& players, player_index_t seat_assignment)
{
  my_index_ = seat_assignment;
  tensorizor_.clear();
  if (!params_.neural_network_only) {
    mcts_.clear();
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline void NNetPlayer<GameState_, Tensorizor_>::receive_state_change(
    player_index_t player, const GameState& state, action_index_t action, const GameResult& result)
{
  tensorizor_.receive_state_change(state, action);
  if (!params_.neural_network_only) {
    mcts_.receive_state_change(player, state, action, result);
  }
  last_action_ = action;
  if (my_index_ == player && params_.verbose) {
    verbose_dump();
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline action_index_t NNetPlayer<GameState_, Tensorizor_>::get_action(const GameState& state, const ActionMask& valid_actions) {
  if (params_.neural_network_only) {
    return get_net_only_action(state, valid_actions);
  } else {
    return get_mcts_action(state, valid_actions);
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline action_index_t NNetPlayer<GameState_, Tensorizor_>::get_net_only_action(
    const GameState& state, const ActionMask& valid_actions)
{
  auto& policy = policy_.asEigen();
  auto& value = value_.asEigen();
  auto& input = input_.asEigen();

  tensorizor_.tensorize(input, state);
  auto transform = tensorizor_.get_symmetry(state, tensorizor_.get_random_symmetry_index(state));
  transform->transform_input(input);
  torch_input_gpu_.copy_(input_.asTorch());
  net_.predict(input_vec_, policy_.asTorch(), value_.asTorch());
  transform->transform_policy(policy);

  value = eigen_util::softmax(value);
  if (verbose_info_) {
    verbose_info_->mcts_results.policy_prior = eigen_util::softmax(policy);
    verbose_info_->mcts_results.value_prior = value;
    verbose_info_->initialized = true;
  }

  if (inv_temperature_) {
    policy = eigen_util::softmax(policy * inv_temperature_);
  } else {
    policy = (policy.array() == policy.maxCoeff()).template cast<float>();
  }
  return util::Random::weighted_sample(policy.begin(), policy.end());
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline action_index_t NNetPlayer<GameState_, Tensorizor_>::get_mcts_action(
    const GameState& state, const ActionMask& valid_actions)
{
  auto& policy = policy_.asEigen();
  auto& value = value_.asEigen();

  auto results = mcts_.sim(tensorizor_, state, mcts_params_);
  policy = results->counts.template cast<float>();
  if (inv_temperature_) {
    policy = policy.array().pow(inv_temperature_);
  } else {
    policy = (policy.array() == policy.maxCoeff()).template cast<float>();
  }

  value = results->win_rates.template cast<float>();
  if (verbose_info_) {
    verbose_info_->mcts_value = value;
    verbose_info_->mcts_policy = policy / policy.sum();
    verbose_info_->mcts_results = *results;
    verbose_info_->initialized = true;
  }
  return util::Random::weighted_sample(policy.begin(), policy.end());
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
inline void NNetPlayer<GameState_, Tensorizor_>::verbose_dump() const {
  if (!verbose_info_->initialized) return;

  const auto& mcts_value = verbose_info_->mcts_value;
  const auto& mcts_policy = verbose_info_->mcts_policy;
  const auto& mcts_results = verbose_info_->mcts_results;

  util::xprintf("CPU pos eval:\n");
  if (params_.neural_network_only) {
    GameState::xdump_nnet_output(mcts_results);
  } else {
    GameState::xdump_mcts_output(mcts_value, mcts_policy, mcts_results);
  }
  util::xprintf("\n");
  util::xflush();
}

}  // namespace common
