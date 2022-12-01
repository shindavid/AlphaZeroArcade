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
  , inv_temperature_(params.temperature ? (1.0 / params.temperature) : 0)
{
  if (!params_.neural_network_only) {
    throw util::Exception("!neural_network_only not yet supported");
  }

  torch_input_ = eigen_util::eigen2torch(input_);
  torch_policy_ = eigen_util::eigen2torch<util::int_sequence<1, PolicyVector::RowsAtCompileTime>>(policy_);
  torch_value_ = eigen_util::eigen2torch<util::int_sequence<1, ValueVector::RowsAtCompileTime>>(value_);

  torch_input_gpu_ = torch_input_.clone().to(torch::kCUDA);
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
  tensorizor_.tensorize(0, input_, state);
  auto transform = tensorizor_.get_random_symmetry(state);
  transform->transform_input(input_);
  torch_input_gpu_.copy_(torch_input_);
  net_.predict(input_vec_, torch_policy_, torch_value_);
  asm volatile("": : :"memory");  // memory barrier - ensures proper read/write order of torch_policy_ and policy_
  transform->transform_policy(policy_);

  value_ = eigen_util::softmax(value_).eval();  // eval() to avoid potential aliasing issue (?)
  if (verbose_info_) {
    verbose_info_->net_value = value_;
    verbose_info_->net_policy = eigen_util::softmax(policy_);
    verbose_info_->initialized = true;
  }

  if (inv_temperature_) {
    policy_ = eigen_util::softmax(policy_* inv_temperature_).eval();  // eval() to avoid potential aliasing issue (?)
  } else {
    policy_ = (policy_.array() == policy_.maxCoeff()).cast<float>();
  }
  return util::Random::weighted_sample(policy_.begin(), policy_.end());
}

inline common::action_index_t NNetPlayer::get_mcts_action(const GameState& state, const ActionMask& valid_actions) {
  auto results = mcts_.sim(tensorizor_, state, mcts_params_);
  policy_ = results->counts.cast<float>();
  if (inv_temperature_) {
    policy_ = policy_.array().pow(inv_temperature_);
  } else {
    policy_ = (policy_.array() == policy_.maxCoeff()).cast<float>();
  }

  value_ = results->win_rates.cast<float>();
  if (verbose_info_) {
    verbose_info_->mcts_value = value_;
    verbose_info_->net_value = results->value_prior;
    verbose_info_->mcts_counts = results->counts;
    verbose_info_->mcts_policy = policy_ / policy_.sum();
    verbose_info_->net_policy = results->policy_prior;
    verbose_info_->initialized = true;
  }
  return util::Random::weighted_sample(policy_.begin(), policy_.end());
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
