#include <connect4/C4NNetPlayer.hpp>

#include <util/Exception.hpp>
#include <util/PrintUtil.hpp>
#include <util/Random.hpp>
#include <util/RepoUtil.hpp>
#include <util/TorchUtil.hpp>

namespace c4 {

inline NNetPlayer::Params::Params()
  : model_filename(util::Repo::root() / "c4_model.pt") {}

inline NNetPlayer::NNetPlayer(const Params& params)
  : Player("CPU")
  , params_(params)
  , net_(params.model_filename)
  , policy_(std::array<int64_t, 2>{1, kNumColumns})
  , value_(std::array<int64_t, 2>{1, kNumPlayers})
  , inv_temperature_(params.temperature ? (1.0 / params.temperature) : 0)
{
  torch_input_gpu_ = input_.asTorch().clone().to(torch::kCUDA);
  input_vec_.push_back(torch_input_gpu_);

  if (params.verbose) {
    verbose_info_ = new VerboseInfo();
  }
}

inline NNetPlayer::~NNetPlayer() {
  if (verbose_info_) {
    delete verbose_info_;
  }
}

inline void NNetPlayer::start_game(const player_array_t& players, common::player_index_t seat_assignment) {
  my_index_ = seat_assignment;
  tensorizor_.clear();
  if (!params_.neural_network_only) {
    mcts_.clear();
  }
}

inline void NNetPlayer::receive_state_change(
    common::player_index_t player, const GameState& state, common::action_index_t action, const Result& result)
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

inline common::action_index_t NNetPlayer::get_action(const GameState& state, const ActionMask& valid_actions) {
  if (params_.neural_network_only) {
    return get_net_only_action(state, valid_actions);
  } else {
    return get_mcts_action(state, valid_actions);
  }
}

inline common::action_index_t NNetPlayer::get_net_only_action(const GameState& state, const ActionMask& valid_actions) {
  auto& policy = policy_.asEigen();
  auto& value = value_.asEigen();
  auto& input = input_.asEigen();

  tensorizor_.tensorize(input, state);
  auto transform = tensorizor_.get_random_symmetry(state);
  transform->transform_input(input);
  torch_input_gpu_.copy_(input_.asTorch());
  net_.predict(input_vec_, policy_.asTorch(), value_.asTorch());
  transform->transform_policy(policy);

  value = eigen_util::softmax(value);
  if (verbose_info_) {
    verbose_info_->net_value = value;
    verbose_info_->net_policy = eigen_util::softmax(policy);
    verbose_info_->initialized = true;
  }

  if (inv_temperature_) {
    policy = eigen_util::softmax(policy * inv_temperature_);
  } else {
    policy = (policy.array() == policy.maxCoeff()).cast<float>();
  }
  return util::Random::weighted_sample(policy.begin(), policy.end());
}

inline common::action_index_t NNetPlayer::get_mcts_action(const GameState& state, const ActionMask& valid_actions) {
  auto& policy = policy_.asEigen();
  auto& value = value_.asEigen();

  auto results = mcts_.sim(tensorizor_, state, mcts_params_);
  policy = results->counts.cast<float>();
  if (inv_temperature_) {
    policy = policy.array().pow(inv_temperature_);
  } else {
    policy = (policy.array() == policy.maxCoeff()).cast<float>();
  }

  value = results->win_rates.cast<float>();
  if (verbose_info_) {
    verbose_info_->mcts_value = value;
    verbose_info_->net_value = results->value_prior;
    verbose_info_->mcts_counts = results->counts;
    verbose_info_->mcts_policy = policy / policy.sum();
    verbose_info_->net_policy = results->policy_prior;
    verbose_info_->initialized = true;
  }
  return util::Random::weighted_sample(policy.begin(), policy.end());
}

inline void NNetPlayer::verbose_dump() const {
  if (!verbose_info_->initialized) return;

  const auto& mcts_value = verbose_info_->mcts_value;
  const auto& net_value = verbose_info_->net_value;
  const auto& mcts_counts = verbose_info_->mcts_counts;
  const auto& mcts_policy = verbose_info_->mcts_policy;
  const auto& net_policy = verbose_info_->net_policy;

  util::xprintf("CPU pos eval:\n");
  if (params_.neural_network_only) {
    util::xprintf("%s%s%s: %6.3f\n", ansi::kRed, ansi::kCircle, ansi::kReset, 100 * net_value(kRed));
    util::xprintf("%s%s%s: %6.3f\n", ansi::kYellow, ansi::kCircle, ansi::kReset, 100 * net_value(kYellow));
    util::xprintf("\n");
    util::xprintf("%3s %8s\n", "Col", "Net");
    for (int i = 0; i < kNumColumns; ++i) {
      util::xprintf("%3d %8.3f\n", i + 1, net_policy(i));
    }
  } else {
    util::xprintf("%s%s%s: %6.3f -> %6.3f\n", ansi::kRed, ansi::kCircle, ansi::kReset, 100 * net_value(kRed),
                  100 * mcts_value(kRed));
    util::xprintf("%s%s%s: %6.3f -> %6.3f\n", ansi::kYellow, ansi::kCircle, ansi::kReset, 100 * net_value(kYellow),
                  100 * mcts_value(kYellow));
    util::xprintf("\n");
    util::xprintf("%3s %8s %8s %8s\n", "Col", "Net", "Count", "MCTS");
    for (int i = 0; i < kNumColumns; ++i) {
      util::xprintf("%3d %8.3f %8d %8.3f\n", i + 1, net_policy(i), mcts_counts(i), mcts_policy(i));
    }
  }
  util::xprintf("\n");
  util::xflush();
}

}  // namespace c4
