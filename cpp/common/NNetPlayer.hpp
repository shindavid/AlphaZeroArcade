#pragma once

#include <ostream>

#include <boost/filesystem.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#include <common/AbstractPlayer.hpp>
#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/Mcts.hpp>
#include <common/MctsResults.hpp>
#include <common/NeuralNet.hpp>
#include <common/TensorizorConcept.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenTorch.hpp>

namespace common {

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
class NNetPlayer : public AbstractPlayer<GameState_> {
public:
  using base_t = AbstractPlayer<GameState_>;
  using GameState = GameState_;
  using Tensorizor = Tensorizor_;

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

  using GameStateTypes = GameStateTypes_<GameState>;
  using TensorizorTypes = TensorizorTypes_<Tensorizor>;

  using Mcts = Mcts_<GameState, Tensorizor>;
  using MctsResults = MctsResults_<GameState>;

  using PolicySlab = typename GameStateTypes::PolicySlab;
  using ValueSlab = typename GameStateTypes::ValueSlab;
  using ActionMask = typename GameStateTypes::ActionMask;
  using GameResult = typename GameStateTypes::GameResult;
  using player_array_t = typename base_t::player_array_t;
  using InputTensor = typename TensorizorTypes::InputTensor;
  using ValueProbDistr = typename Mcts::ValueProbDistr;
  using LocalPolicyProbDistr = typename Mcts::LocalPolicyProbDistr;

  NNetPlayer(const Params&);
  ~NNetPlayer();

  void start_game(const player_array_t& players, player_index_t seat_assignment) override;
  void receive_state_change(player_index_t, const GameState&, action_index_t, const GameResult&) override;
  action_index_t get_action(const GameState&, const ActionMask&) override;

private:
  struct VerboseInfo {
    ValueProbDistr mcts_value;
    LocalPolicyProbDistr mcts_policy;
    MctsResults mcts_results;

    bool initialized = false;
  };

  action_index_t get_net_only_action(const GameState&, const ActionMask&);
  action_index_t get_mcts_action(const GameState&, const ActionMask&);
  void verbose_dump() const;

  const Params& params_;
  NeuralNet net_;
  Tensorizor tensorizor_;

  PolicySlab policy_;
  ValueSlab value_;
  InputTensor input_;

  NeuralNet::input_vec_t input_vec_;
  torch::Tensor torch_input_gpu_;

  Mcts mcts_;
  typename Mcts::Params mcts_params_;
  const float inv_temperature_;
  action_index_t last_action_ = -1;
  player_index_t my_index_ = -1;
  VerboseInfo* verbose_info_ = nullptr;
};

}  // namespace common

#include <common/inl/NNetPlayer.inl>
