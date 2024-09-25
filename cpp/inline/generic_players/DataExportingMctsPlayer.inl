#pragma once

#include <generic_players/DataExportingMctsPlayer.hpp>

#include <util/BitSet.hpp>

namespace generic {

template <core::concepts::Game Game>
template <typename... BaseArgs>
DataExportingMctsPlayer<Game>::DataExportingMctsPlayer(
    const TrainingDataWriterParams& writer_params, BaseArgs&&... base_args)
    : base_t(std::forward<BaseArgs>(base_args)...),
      writer_(TrainingDataWriter::instantiate(writer_params)) {}

template <core::concepts::Game Game>
void DataExportingMctsPlayer<Game>::start_game() {
  base_t::start_game();
  game_log_ = writer_->get_log(base_t::get_game_id());
}

template <core::concepts::Game Game>
void DataExportingMctsPlayer<Game>::receive_state_change(core::seat_index_t seat,
                                                         const State& state,
                                                         core::action_t action) {
  base_t::receive_state_change(seat, state, action);
}

template <core::concepts::Game Game>
core::ActionResponse DataExportingMctsPlayer<Game>::get_action_response(
    const State& state, const ActionMask& valid_actions) {
  auto search_mode = this->choose_search_mode();
  bool use_for_training = search_mode == core::kFull;
  bool previous_used_for_training = game_log_->is_previous_entry_used_for_training();

  if (kForceFullSearchIfRecordingAsOppReply && previous_used_for_training) {
    search_mode = core::kFull;
  }

  const SearchResults* mcts_search_results = this->mcts_search(search_mode);

  PolicyTensor policy_target;
  PolicyTensor* policy_target_ptr = nullptr;
  if (use_for_training || previous_used_for_training) {
    policy_target_ptr = &policy_target;
    extract_policy_target(mcts_search_results, &policy_target_ptr);
  }
  ActionValueTensor action_values;
  ActionValueTensor* action_values_ptr = nullptr;
  if (use_for_training) {
    action_values = mcts_search_results->action_values;
    action_values_ptr = &action_values;
  }
  core::ActionResponse response =
      base_t::get_action_response_helper(search_mode, mcts_search_results, valid_actions);
  game_log_->add(state, response.action, policy_target_ptr, action_values_ptr, use_for_training);
  return response;
}

template <core::concepts::Game Game>
void DataExportingMctsPlayer<Game>::end_game(const State& state,
                                             const ValueArray& outcome) {
  game_log_->add_terminal(state, outcome);
  writer_->close(game_log_);
  game_log_ = nullptr;
}

template <core::concepts::Game Game>
void DataExportingMctsPlayer<Game>::extract_policy_target(const SearchResults* mcts_results,
                                                          PolicyTensor** target) {
  **target = mcts_results->policy_target;
  float sum = eigen_util::sum(**target);
  if (mcts_results->provably_lost || sum == 0 || mcts_results->trivial) {
    // python training code will ignore these rows for policy training.
    *target = nullptr;
  } else {
    **target = mcts_results->action_symmetry_table.symmetrize(**target);
    sum = eigen_util::sum(**target);
    auto& policy_target_array = eigen_util::reinterpret_as_array(**target);
    policy_target_array /= sum;
  }
}

}  // namespace generic
