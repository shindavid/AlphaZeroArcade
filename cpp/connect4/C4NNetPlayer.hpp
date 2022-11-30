#pragma once

#include <ostream>

#include <boost/filesystem.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/Mcts.hpp>
#include <common/NeuralNet.hpp>
#include <connect4/C4Constants.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <util/CppUtil.hpp>

namespace c4 {

class NNetPlayer : public Player {
public:
  struct Params {
    Params();

    boost::filesystem::path model_filename;
    boost::filesystem::path debug_filename;
    bool verbose = false;
    bool neural_network_only = false;
    bool allow_elimination = true;
    int num_mcts_iters = 100;
    float temperature = 0;
  };

  using Mcts = common::Mcts<GameState, Tensorizor>;
  using PolicyVector = common::GameStateTypes<GameState>::PolicyVector;
  using ValueVector = common::GameStateTypes<GameState>::ValueVector;
  using InputTensor = common::TensorizorTypes<Tensorizor>::InputTensor;

  NNetPlayer(const Params&);
  void start_game(const player_array_t& players, common::player_index_t seat_assignment) override;
  void receive_state_change(common::player_index_t, const GameState&, common::action_index_t, const Result&) override;
  common::action_index_t get_action(const GameState&, const ActionMask&) override;

private:
  struct VerboseInfo {
    Mcts::ValueProbDistr value;
    Mcts::ValueProbDistr mcts_value;
    Mcts::ValueProbDistr net_value;

    // TODO: Make all these local
    Mcts::GlobalPolicyProbDistr policy;
    Mcts::GlobalPolicyProbDistr mcts_policy;
    Mcts::GlobalPolicyCountDistr  mcts_counts;
    Mcts::GlobalPolicyProbDistr net_policy;
  };

  common::action_index_t get_net_only_action(const GameState&, const ActionMask&);
  common::action_index_t get_mcts_action(const GameState&, const ActionMask&);
  common::action_index_t get_action_helper();
  void verbose_dump() const;

  const Params& params_;
  common::NeuralNet net_;
  Tensorizor tensorizor_;

  /*
   * torch_input_ is backed by input_
   * torch_policy_ is backed by policy_
   * torch_value_ is backed by value_
   *
   * These backings allow us to use Eigen API's to modify the tensors. The Eigen objects have compile-time-known
   * dtypes/dimensions and the torch counterparts do not. Doing modifications on the Eigen objects thus allows for more
   * efficient modification operations, as well as better control over memory allocations.
   */
  torch::Tensor torch_input_, torch_policy_, torch_value_;
  InputTensor input_;
  PolicyVector policy_;
  ValueVector value_;

  common::NeuralNet::input_vec_t input_vec_;
  torch::Tensor torch_input_gpu_;

  Mcts mcts_;
  Mcts::Params mcts_params_;
  const float inv_temperature_;
  common::action_index_t last_action_ = -1;
  common::player_index_t my_index_ = -1;
};

}  // namespace c4

#include <connect4/inl/C4NNetPlayer.inl>
