#pragma once

#include <generic_players/DataExportingMctsPlayer.hpp>

#include <util/BitSet.hpp>

namespace generic {

template <core::concepts::Game Game>
core::ActionResponse DataExportingMctsPlayer<Game>::get_action_response(
    const State& state, const ActionMask& valid_actions) {
  auto search_mode = this->choose_search_mode();
  bool use_for_training = search_mode == core::kFull;

  GameLogWriter_sptr game_log = this->get_game_log();
  bool previous_used_for_training =
      game_log && game_log->was_previous_entry_used_for_policy_training();

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

  // TODO: augment response with policy_target/action_values_target/use_for_training
  throw std::runtime_error("Not implemented");

  // return response;
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
