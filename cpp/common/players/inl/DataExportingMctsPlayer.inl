#pragma once

#include <common/players/DataExportingMctsPlayer.hpp>

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
    seat_index_t seat, const GameState& state, action_index_t action)
{
  base_t::receive_state_change(seat, state, action);
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
    game_data_->record_for_all(outcome);
    writer_->close(game_data_);
    game_data_ = nullptr;
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void DataExportingMctsPlayer<GameState_, Tensorizor_>::record_position(
    const GameState& state, const MctsResults* mcts_results)
{
  auto group = game_data_->get_next_group();
  auto& input = group.input;
  auto& policy = group.policy;
  auto& current_player = group.current_player;

  this->tensorizor_.tensorize(input, state);

  policy = mcts_results->counts;
  auto& policy_array = eigen_util::reinterpret_as_array(policy);
  float sum = policy_array.sum();
  if (sum == 0) {
    // Happens if eliminations is enabled and MCTS proves that the position is losing.
    policy_array.setConstant(1.0);
    sum = policy_array.sum();
  }

  policy_array /= sum;
  current_player = this->get_my_seat();
}

}  // namespace common
