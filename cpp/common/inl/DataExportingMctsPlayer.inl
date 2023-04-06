#pragma once

#include <common/DataExportingMctsPlayer.hpp>

#include <util/BitSet.hpp>

namespace common {

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
template<typename... BaseArgs>
DataExportingMctsPlayer<GameState_, Tensorizor_>::DataExportingMctsPlayer(
    const TrainingDataWriterParams& writer_params, BaseArgs&&... base_args)
: base_t(std::forward<BaseArgs>(base_args)...)
, writer_(TrainingDataWriter::instantiate(writer_params)) {}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void DataExportingMctsPlayer<GameState_, Tensorizor_>::start_game()
{
  base_t::start_game();
  game_data_ = writer_->get_data(base_t::get_game_id());
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void DataExportingMctsPlayer<GameState_, Tensorizor_>::receive_state_change(
    player_index_t p, const GameState& state, action_index_t action)
{
  base_t::receive_state_change(p, state, action);
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
action_index_t DataExportingMctsPlayer<GameState_, Tensorizor_>::get_action(
    const GameState& state, const ActionMask& valid_actions)
{
  auto search_mode = this->choose_search_mode();
  const MctsResults* mcts_results = this->mcts_search(state, search_mode);

  if (search_mode == base_t::kFull) {
    record_position(state, mcts_results);
  }
  return base_t::get_action_helper(search_mode, mcts_results, valid_actions);
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void DataExportingMctsPlayer<GameState_, Tensorizor_>::end_game(const GameState&, const GameOutcome& outcome) {
  if (is_terminal_outcome(outcome)) {
    game_data_->record_for_all(outcome.reshaped(1, GameState::kNumPlayers));
    writer_->close(game_data_);
    game_data_ = nullptr;
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void DataExportingMctsPlayer<GameState_, Tensorizor_>::record_position(
    const GameState& state, const MctsResults* mcts_results)
{
  auto slab = game_data_->get_next_slab();
  auto& input = slab.input;
  auto& policy = slab.policy;

  this->tensorizor_.tensorize(input, state);

  GlobalPolicyProbDistr counts = mcts_results->counts.reshaped(1, GameState::kNumGlobalActions);
  float sum = counts.sum();
  if (sum == 0) {
    // Happens if eliminations is enabled and MCTS proves that the position is losing.
    counts.setConstant(1.0);
    sum = counts.sum();
  }

  policy = counts / sum;
}

}  // namespace common
