#pragma once

#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/TensorizorConcept.hpp>
#include <common/TrainingDataWriter.hpp>
#include <common/players/MctsPlayer.hpp>

namespace common {

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
class DataExportingMctsPlayer : public MctsPlayer<GameState_, Tensorizor_> {
public:
  using GameState = GameState_;
  using Tensorizor = Tensorizor_;
  using GameStateTypes = common::GameStateTypes<GameState>;
  using TensorizorTypes = common::TensorizorTypes<Tensorizor>;
  using dtype = typename GameStateTypes::dtype;
  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using TrainingDataWriter = common::TrainingDataWriter<GameState, Tensorizor>;
  using TrainingDataWriterParams = typename TrainingDataWriter::Params;
  using InputTensor = typename TensorizorTypes::InputTensor;

  using base_t = MctsPlayer<GameState, Tensorizor>;
  using Params = base_t::Params;
  using Mcts = base_t::Mcts;
  using MctsResults = base_t::MctsResults;

  template<typename... BaseArgs>
  DataExportingMctsPlayer(const TrainingDataWriterParams& writer_params, BaseArgs&&...);

  void start_game() override;
  void receive_state_change(
      seat_index_t seat, const GameState& state, action_index_t action) override;
  action_index_t get_action(const GameState&, const ActionMask&) override;
  void end_game(const GameState&, const GameOutcome&) override;

protected:
  void record_position(const GameState& state, const ActionMask& valid_actions, const MctsResults*);

  TrainingDataWriter* writer_;
  TrainingDataWriter::GameData_sptr game_data_;
};

}  // namespace common

#include <common/players/inl/DataExportingMctsPlayer.inl>

