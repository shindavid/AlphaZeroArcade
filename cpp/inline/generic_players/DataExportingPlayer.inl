
#include "generic_players/DataExportingPlayer.hpp"

#include "core/BasicTypes.hpp"
#include "core/Constants.hpp"

namespace generic {

template <typename BasePlayer>
core::yield_instruction_t DataExportingPlayer<BasePlayer>::handle_chance_event(
  const ChanceEventHandleRequest& request) {
  if (!game_log_) return core::kContinue;
  if (!this->owns_shared_data_) return core::kContinue;  // so only one player handles chance events

  if (!mid_handle_chance_event_) {
    // Sample chance events at the same frequency as we do for player events. This seems right, as
    // it ensures that chance events are represented in the training data proportionally to how
    // often they occur in the game.
    if (this->get_random_search_mode() != core::kFull) return core::kContinue;
    mid_handle_chance_event_ = true;
    training_info_.clear();
  }

  core::yield_instruction_t instruction =
    this->get_manager()->load_root_action_values(request, this->get_my_seat(), training_info_);

  if (instruction == core::kContinue) {
    mid_handle_chance_event_ = false;
    game_log_->add(training_info_);
  }
  return instruction;
}

template <typename BasePlayer>
bool DataExportingPlayer<BasePlayer>::start_game() {
  BasePlayer::start_game();
  if (writer_) {
    if (writer_->closed()) return false;
    game_log_ = writer_->get_game_log(this->get_game_id());
  }
  return true;
}

template <typename BasePlayer>
void DataExportingPlayer<BasePlayer>::end_game(const State& state, const ValueTensor& outcome) {
  BasePlayer::end_game(state, outcome);
  if (!game_log_) return;
  game_log_->add_terminal(state, outcome);  // redundant if multiple players, but that's ok
  writer_->add(game_log_);
}

template <typename BasePlayer>
DataExportingPlayer<BasePlayer>::ActionResponse
DataExportingPlayer<BasePlayer>::get_action_response_helper(const SearchResults* mcts_results,
                                                            const ActionRequest& request) {
  ActionResponse response = BasePlayer::get_action_response_helper(mcts_results, request);
  add_to_game_log(request, response, mcts_results);

  return response;
}

template <typename BasePlayer>
void DataExportingPlayer<BasePlayer>::add_to_game_log(const ActionRequest& request,
                                                      const ActionResponse& response,
                                                      const SearchResults* mcts_results) {
  if (!game_log_) return;

  bool use_for_training = this->search_mode_ == core::kFull;

  // TODO: if we have chance-events between player-events, we should compute this bool
  // differently.
  bool previous_used_for_training = game_log_->was_previous_entry_used_for_policy_training();

  training_info_.clear();
  core::seat_index_t my_seat = this->get_my_seat();

  TrainingInfoParams params;
  params.state = request.state;
  params.mcts_results = mcts_results;
  params.action = response.action;
  params.seat = my_seat;
  params.use_for_training = use_for_training;
  params.previous_used_for_training = previous_used_for_training;

  Algorithms::write_to_training_info(params, training_info_);

  game_log_->add(training_info_);
}

template <typename BasePlayer>
void DataExportingPlayer<BasePlayer>::extract_policy_target(const SearchResults* mcts_results) {
  auto& target = training_info_.policy_target;
  target = mcts_results->policy_target;

  float sum = eigen_util::sum(target);
  if (mcts_results->provably_lost || sum == 0 || mcts_results->trivial) {
    // python training code will ignore these rows for policy training.
    training_info_.policy_target_valid = false;
  } else {
    target = mcts_results->action_symmetry_table.symmetrize(target);
    target = target / eigen_util::sum(target);
  }
}

}  // namespace generic
