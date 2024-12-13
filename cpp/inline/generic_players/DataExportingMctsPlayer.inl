#pragma once

#include <generic_players/DataExportingMctsPlayer.hpp>

#include <util/BitSet.hpp>

namespace generic {

template <core::concepts::Game Game>
typename DataExportingMctsPlayer<Game>::ActionResponse
DataExportingMctsPlayer<Game>::get_action_response(const State& state,
                                                   const ActionMask& valid_actions) {
  auto search_mode = this->choose_search_mode();

  GameLogWriter_sptr game_log = this->get_game_log();
  bool use_for_training = game_log && search_mode == core::kFull;
  bool previous_used_for_training =
      game_log && game_log->was_previous_entry_used_for_policy_training();

  if (kForceFullSearchIfRecordingAsOppReply && previous_used_for_training) {
    search_mode = core::kFull;
  }

  const SearchResults* mcts_search_results = this->mcts_search(search_mode);
  ActionResponse response =
      base_t::get_action_response_helper(search_mode, mcts_search_results, valid_actions);

  TrainingInfo& training_info = response.training_info;
  training_info.policy_target = nullptr;
  training_info.action_values_target = nullptr;
  training_info.use_for_training = use_for_training;

  if (use_for_training || previous_used_for_training) {
    training_info.policy_target = &policy_target_;
    extract_policy_target(mcts_search_results, &training_info.policy_target);
  }
  if (use_for_training) {
    action_values_target_ = mcts_search_results->action_values;
    training_info.action_values_target = &action_values_target_;
  }

  return response;
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
    **target = **target / eigen_util::sum(**target);
  }
}

}  // namespace generic
