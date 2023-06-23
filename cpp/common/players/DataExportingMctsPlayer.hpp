#pragma once

#include <common/players/MctsPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <core/TrainingDataWriter.hpp>
#include <mcts/SearchResults.hpp>

#include <vector>

namespace common {

/*
 * A variant of MctsPlayer that exports training data to a file via TrainingDataWriter.
 */
template<core::GameStateConcept GameState_, core::TensorizorConcept<GameState_> Tensorizor_>
class DataExportingMctsPlayer : public MctsPlayer<GameState_, Tensorizor_> {
public:
  /*
   * The argument for using a full search is so that the opp reply target is more accurate.
   *
   * The argument against is that the opp reply target is not that important, making full searches for that purpose
   * an inefficient use of compute budget.
   */
  static constexpr bool kForceFullSearchIfRecordingAsOppReply = false;

  using GameState = GameState_;
  using Tensorizor = Tensorizor_;
  using GameStateTypes = core::GameStateTypes<GameState>;
  using TensorizorTypes = core::TensorizorTypes<Tensorizor>;
  using dtype = typename GameStateTypes::dtype;
  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using TrainingDataWriter = core::TrainingDataWriter<GameState, Tensorizor>;
  using TrainingDataWriterParams = typename TrainingDataWriter::Params;
  using InputTensor = typename TensorizorTypes::InputTensor;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;

  using base_t = MctsPlayer<GameState, Tensorizor>;
  using Params = base_t::Params;
  using MctsManager = base_t::MctsManager;
  using MctsSearchResults = base_t::MctsSearchResults;

  template<typename... BaseArgs>
  DataExportingMctsPlayer(const TrainingDataWriterParams& writer_params, BaseArgs&&...);

  void start_game() override;
  void receive_state_change(
      core::seat_index_t seat, const GameState& state, core::action_index_t action) override;
  core::action_index_t get_action(const GameState&, const ActionMask&) override;
  void end_game(const GameState&, const GameOutcome&) override;

protected:
  static PolicyTensor extract_policy(const MctsSearchResults* results);
  void record_position(const GameState& state, const ActionMask& valid_actions, const PolicyTensor& policy);

  TrainingDataWriter* writer_;
  TrainingDataWriter::GameData_sptr game_data_;
};

}  // namespace common

#include <common/players/inl/DataExportingMctsPlayer.inl>

