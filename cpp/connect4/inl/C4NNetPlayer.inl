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
  , inv_temperature_(params.temperature ? (1.0 / params.temperature) : 0)
{
  if (!params_.neural_network_only) {
    throw util::Exception("!neural_network_only not yet supported");
  }

  torch_input_ = eigen_util::eigen2torch(input_);
  torch_policy_ = eigen_util::eigen2torch(policy_);
  torch_value_ = eigen_util::eigen2torch(value_);

  torch_input_gpu_ = torch_input_.clone().to(torch::kCUDA);
  input_vec_.push_back(torch_input_gpu_);
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
  torch_input_gpu_ = torch_input_;
  net_.predict(input_vec_, torch_policy_, torch_value_);
  transform->transform_policy(policy_);

  if (params_.temperature) {
//    policy_ = policy_ * inv_temperature_;
    policy_ = eigen_util::softmax(policy_* inv_temperature_).eval();  // eval() to avoid potential aliasing issue (?)
  } else {
    throw std::exception();  // TODO
//    policy_ = (policy_ == policy_.maximum());
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
