#pragma once

#include <common/AbstractPlayer.hpp>
#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/TensorizorConcept.hpp>
#include <common/TrainingDataWriter.hpp>

namespace common {

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_, typename Player_>
class DataExportingPlayer : public Player_ {
public:
  using GameState = GameState_;
  using Tensorizor = Tensorizor_;
  using GameStateTypes = common::GameStateTypes<GameState>;
  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using GlobalPolicyCountDistr = typename GameStateTypes::GlobalPolicyCountDistr;
  using TrainingDataWriter = common::TrainingDataWriter<GameState, Tensorizor>;

  using base_t = Player_;
  using player_array_t = AbstractPlayer<GameState>::player_array_t;

  template<typename... BaseArgTs>
  DataExportingPlayer(TrainingDataWriter* writer, BaseArgTs&&...);

  void start_game(game_id_t, const player_array_t& players, player_index_t seat_assignment) override;
  void receive_state_change(
      player_index_t p, const GameState& state, action_index_t action,
      const GameOutcome& outcome) override;

  action_index_t get_action(const GameState&, const ActionMask&) override;

protected:
  void record_position(const GameState& state);

  TrainingDataWriter* writer_;
  TrainingDataWriter::GameData_sptr game_data_;
};

}  // namespace common

#include <common/inl/DataExportingPlayer.inl>
