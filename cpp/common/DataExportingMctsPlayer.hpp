#pragma once

#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/MctsPlayer.hpp>
#include <common/TensorizorConcept.hpp>
#include <common/TrainingDataWriter.hpp>

namespace common {

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
class DataExportingMctsPlayer : public MctsPlayer<GameState_, Tensorizor_> {
public:
  using GameState = GameState_;
  using Tensorizor = Tensorizor_;
  using GameStateTypes = common::GameStateTypes<GameState>;
  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using GlobalPolicyProbDistr = typename GameStateTypes::GlobalPolicyProbDistr;
  using TrainingDataWriter = common::TrainingDataWriter<GameState, Tensorizor>;
  using TrainingDataWriterParams = typename TrainingDataWriter::Params;

  using base_t = MctsPlayer<GameState, Tensorizor>;
  using Params = base_t::Params;
  using Mcts = base_t::Mcts;
  using MctsResults = base_t::MctsResults;
  using player_array_t = base_t::player_array_t;

  template<typename... BaseArgs>
  DataExportingMctsPlayer(const TrainingDataWriterParams& writer_params, BaseArgs&&...);

  void start_game() override;
  void receive_state_change(
      player_index_t p, const GameState& state, action_index_t action,
      const GameOutcome& outcome) override;

  action_index_t get_action(const GameState&, const ActionMask&) override;

protected:
  void record_position(const GameState& state, const MctsResults*);

  TrainingDataWriter* writer_;
  TrainingDataWriter::GameData_sptr game_data_;
};

}  // namespace common

#include <common/inl/DataExportingMctsPlayer.inl>
