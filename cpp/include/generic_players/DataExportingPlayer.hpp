#pragma once

#include "core/TrainingDataWriter.hpp"

namespace generic {

/*
 * A generic player that exports training data to a file via TrainingDataWriter.
 *
 * Assumes that BasePlayer is either one of the following:
 *
 * - generic::alpha0::Player<Traits>
 * - generic::beta0::Player<Traits>
 */
template <typename BasePlayer>
class DataExportingPlayer : public BasePlayer {
 public:
  using Traits = BasePlayer::Traits;
  using Game = BasePlayer::Game;
  using State = Game::State;
  using ActionMask = Game::Types::ActionMask;
  using ValueTensor = Game::Types::ValueTensor;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using ActionRequest = Game::Types::ActionRequest;
  using ActionResponse = Game::Types::ActionResponse;
  using ChangeEventHandleRequest = Game::Types::ChangeEventHandleRequest;
  using TrainingInfo = Game::Types::TrainingInfo;

  using SearchResults = BasePlayer::SearchResults;
  using SearchResponse = BasePlayer::SearchResponse;

  using TrainingDataWriter = core::TrainingDataWriter<Game>;
  using GameWriteLog = TrainingDataWriter::GameWriteLog;
  using GameWriteLog_sptr = TrainingDataWriter::GameWriteLog_sptr;

  template <typename... Ts>
  DataExportingPlayer(Ts&&... args)
      : BasePlayer(std::forward<Ts>(args)...), writer_(TrainingDataWriter::instance()) {}

  core::yield_instruction_t handle_chance_event(const ChangeEventHandleRequest&) override;
  bool start_game() override;
  void end_game(const State&, const ValueTensor&) override;

 protected:
  ActionResponse get_action_response_helper(const SearchResults*, const ActionRequest&) override;

  void add_to_game_log(const ActionRequest&, const ActionResponse&, const SearchResults*);
  void extract_policy_target(const SearchResults* results);

  TrainingDataWriter* const writer_;
  TrainingInfo training_info_;
  GameWriteLog_sptr game_log_;
  bool mid_handle_chance_event_ = false;
};

}  // namespace generic

#include "inline/generic_players/DataExportingPlayer.inl"
