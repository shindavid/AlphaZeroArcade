#pragma once

#include <common/players/DataExportingMctsPlayer.hpp>

#include <util/BitSet.hpp>

namespace common {

template<core::GameStateConcept GameState_, core::TensorizorConcept<GameState_> Tensorizor_>
template<typename... BaseArgs>
DataExportingMctsPlayer<GameState_, Tensorizor_>::DataExportingMctsPlayer(
    const TrainingDataWriterParams& writer_params, BaseArgs&&... base_args)
: base_t(std::forward<BaseArgs>(base_args)...)
, writer_(TrainingDataWriter::instantiate(writer_params)) {}

template<core::GameStateConcept GameState_, core::TensorizorConcept<GameState_> Tensorizor_>
void DataExportingMctsPlayer<GameState_, Tensorizor_>::start_game()
{
  base_t::start_game();
  game_data_ = writer_->get_data(base_t::get_game_id());
}

template<core::GameStateConcept GameState_, core::TensorizorConcept<GameState_> Tensorizor_>
void DataExportingMctsPlayer<GameState_, Tensorizor_>::receive_state_change(
    core::seat_index_t seat, const GameState& state, core::action_t action)
{
  base_t::receive_state_change(seat, state, action);
}

template<core::GameStateConcept GameState_, core::TensorizorConcept<GameState_> Tensorizor_>
core::action_t DataExportingMctsPlayer<GameState_, Tensorizor_>::get_action(
    const GameState& state, const ActionMask& valid_actions)
{
  auto search_mode = this->choose_search_mode();
  bool record = search_mode == base_t::kFull;
  bool record_reply = game_data_->contains_pending_groups();

  if (kForceFullSearchIfRecordingAsOppReply && record_reply) {
    search_mode = base_t::kFull;
  }

  const MctsSearchResults* mcts_search_results = this->mcts_search(state, search_mode);

  if (record_reply || record) {
    auto policy = extract_policy(mcts_search_results);
    if (record_reply) {
      game_data_->commit_opp_reply_to_pending_groups(policy);
    }

    if (record) {
      record_position(state, valid_actions, policy);
    }
  }

  return base_t::get_action_helper(search_mode, mcts_search_results, valid_actions);
}

template<core::GameStateConcept GameState_, core::TensorizorConcept<GameState_> Tensorizor_>
void DataExportingMctsPlayer<GameState_, Tensorizor_>::end_game(const GameState&, const GameOutcome& outcome) {
  game_data_->record_for_all(outcome);
  writer_->close(game_data_);
  game_data_ = nullptr;
}

template<core::GameStateConcept GameState_, core::TensorizorConcept<GameState_> Tensorizor_>
DataExportingMctsPlayer<GameState_, Tensorizor_>::PolicyTensor
DataExportingMctsPlayer<GameState_, Tensorizor_>::extract_policy(const MctsSearchResults* mcts_results) {
  auto policy = mcts_results->counts;
  auto& policy_array = eigen_util::reinterpret_as_array(policy);
  float sum = policy_array.sum();
  if (mcts_results->provably_lost || sum == 0) {
    // python training code will ignore these rows for policy training.
    policy.setZero();
  } else {
    policy_array /= sum;
  }
  return policy;
}

template<core::GameStateConcept GameState_, core::TensorizorConcept<GameState_> Tensorizor_>
void DataExportingMctsPlayer<GameState_, Tensorizor_>::record_position(
    const GameState& state, const ActionMask& valid_actions, const PolicyTensor& policy)
{
  InputTensor input;
  this->tensorizor_.tensorize(input, state);

  using hash_array_t = std::array<uint64_t, Tensorizor::kMaxNumSymmetries>;
  hash_array_t hashes = {};

  auto sym_indices = this->tensorizor_.get_symmetry_indices(state);
  for (core::symmetry_index_t sym_index : bitset_util::on_indices(sym_indices)) {
    auto transform = this->tensorizor_.get_symmetry(sym_index);

    InputTensor sym_input = input;
    transform->transform_input(sym_input);
    uint64_t hash = eigen_util::hash(sym_input);
    hashes[sym_index] = hash;
    bool clash = false;
    for (int s = 0; s < sym_index; ++s) {
      if (hashes[s] == hash) {
        clash = true;
        break;
      }
    }

    if (clash) continue;

    auto& group = game_data_->get_next_group();

    group.input = sym_input;
    group.policy = policy;
    group.opp_policy.setZero();
    group.current_player = this->get_my_seat();

    transform->transform_policy(group.policy);
    game_data_->add_pending_group(transform, &group);
  }
}

}  // namespace common
