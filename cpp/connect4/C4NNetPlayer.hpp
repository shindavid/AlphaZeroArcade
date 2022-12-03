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
#include <util/EigenTorch.hpp>

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

  using EigenTorchPolicy = eigentorch::to_eigentorch_t<PolicyVector>;
  using EigenTorchValue = eigentorch::to_eigentorch_t<ValueVector>;
  using EigenTorchInput = eigentorch::to_eigentorch_t<InputTensor>;

  NNetPlayer(const Params&);
  ~NNetPlayer();

  void start_game(const player_array_t& players, common::player_index_t seat_assignment) override;
  void receive_state_change(common::player_index_t, const GameState&, common::action_index_t, const Result&) override;
  common::action_index_t get_action(const GameState&, const ActionMask&) override;

private:
  struct VerboseInfo {
    Mcts::ValueProbDistr mcts_value;
    Mcts::ValueProbDistr net_value;

    // TODO: Make all these local
    Mcts::GlobalPolicyCountDistr mcts_counts;
    Mcts::GlobalPolicyProbDistr mcts_policy;
    Mcts::GlobalPolicyProbDistr net_policy;

    bool initialized = false;
  };

  common::action_index_t get_net_only_action(const GameState&, const ActionMask&);
  common::action_index_t get_mcts_action(const GameState&, const ActionMask&);
  void verbose_dump() const;

  const Params& params_;
  common::NeuralNet net_;
  Tensorizor tensorizor_;

  EigenTorchPolicy policy_;
  EigenTorchValue value_;
  EigenTorchInput input_;

  common::NeuralNet::input_vec_t input_vec_;
  torch::Tensor torch_input_gpu_;

  Mcts mcts_;
  Mcts::Params mcts_params_;
  const float inv_temperature_;
  common::action_index_t last_action_ = -1;
  common::player_index_t my_index_ = -1;
  VerboseInfo* verbose_info_ = nullptr;
};

}  // namespace c4

#include <connect4/inl/C4NNetPlayer.inl>
