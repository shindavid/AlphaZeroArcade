#include <connect4/C4NNetPlayer.hpp>

#include <util/Exception.hpp>
#include <util/RepoUtil.hpp>
#include <util/TorchUtil.hpp>

namespace c4 {

inline NNetPlayer::Params::Params()
  : model_filename(util::Repo::root() / "c4_model.pt") {}

inline NNetPlayer::NNetPlayer(const Params& params)
  : Player("CPU")
  , params_(params)
  , net_(params.model_filename)
{
  if (!params_.neural_network_only) {
    throw util::Exception("!neural_network_only not yet supported");
  }
  torch_util::init_tensor(policy_);
  torch_util::init_tensor(value_);
  input_ = torch::zeros(util::std_array_v<int64_t, TensorShape>).to(at::kCUDA);
  input_vec_.push_back(input_);
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
  CATCH_TENSOR_MALLOCS(input_);
  CATCH_TENSOR_MALLOCS(policy_);
  CATCH_TENSOR_MALLOCS(value_);
  tensorizor_.tensorize(input_[0], state);
  auto transform = tensorizor_.get_random_symmetry(state);
  transform->transform_input(input_[0]);

  net_.predict(input_vec_, policy_, value_);
  transform->transform_policy(policy_);

  if (params_.temperature) {
    policy_ /= params_.temperature;
    torch::softmax_out(policy_, policy_, 0);
  } else {
    // TODO: I think there is multiple dynamic memory allocation in below
    // 2 lines. Fix me!
    policy_ -= torch::max(policy_);
    policy_ = (policy_ >= 0).to(torch::kFloat);
  }
  throw std::exception();
}

inline common::action_index_t NNetPlayer::get_mcts_action(const GameState& state, const ActionMask& valid_actions) {
  throw std::exception();
}

inline common::action_index_t NNetPlayer::get_action_helper() {
  throw std::exception();
}

inline void NNetPlayer::verbose_dump() const {
  throw std::exception();
}

}  // namespace c4
