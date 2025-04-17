#pragma once

#include <generic_players/DataExportingMctsPlayer.hpp>

#include <util/BitSet.hpp>

namespace generic {

template <core::concepts::Game Game>
typename DataExportingMctsPlayer<Game>::ActionResponse
DataExportingMctsPlayer<Game>::get_action_response(const ActionRequest& request) {
  const ActionMask& valid_actions = request.valid_actions;

  std::unique_lock lock(this->search_mode_mutex_);
  if (this->init_search_mode(request)) {
    GameWriteLog_sptr game_log = this->get_game_log();
    use_for_training_ = game_log && this->search_mode_ == core::kFull;

    // TODO: if we have chance-events between player-events, we should compute this bool
    // differently.
    previous_used_for_training_ =
        game_log && game_log->was_previous_entry_used_for_policy_training();

    if (kForceFullSearchIfRecordingAsOppReply && previous_used_for_training_) {
      this->search_mode_ = core::kFull;
    }
  }
  lock.unlock();

  const SearchResults* mcts_results = this->get_manager()->search();
  if (!mcts_results) {
    return ActionResponse::yield();
  }

  ActionResponse response =
      base_t::get_action_response_helper(mcts_results, valid_actions);

  TrainingInfo& training_info = response.training_info;
  training_info.policy_target = nullptr;
  training_info.action_values_target = nullptr;
  training_info.use_for_training = use_for_training_;

  if (use_for_training_ || previous_used_for_training_) {
    training_info.policy_target = &policy_target_;
    extract_policy_target(mcts_results, &training_info.policy_target);
  }
  if (use_for_training_) {
    action_values_target_ = mcts_results->action_values;
    training_info.action_values_target = &action_values_target_;
  }

  return response;
}

template <core::concepts::Game Game>
typename DataExportingMctsPlayer<Game>::ActionValueTensor*
DataExportingMctsPlayer<Game>::prehandle_chance_event() {
  // So that only one player outputs the action values.
  if (!this->owns_shared_data_) {
    return nullptr;
  }

  // Sample chance events at the same frequency as we do for player events. This seems right, as it
  // ensures that chance events are represented in the training data proportionally to how often
  // they occur in the game.
  if (this->get_random_search_mode() != core::kFull) {
    return nullptr;
  }

  this->get_manager()->load_root_action_values(action_values_target_);
  return &action_values_target_;
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
