#pragma once

#include "core/ActionRequest.hpp"
#include "core/ActionResponse.hpp"
#include "core/ChanceEventHandleRequest.hpp"
#include "search/TrainingDataWriter.hpp"

namespace generic {

/*
 * A generic player that exports training data to a file via TrainingDataWriter.
 *
 * Assumes that BasePlayer is either one of the following:
 *
 * - generic::alpha0::Player<Spec>
 */
template <typename BasePlayer_>
class DataExportingPlayer : public BasePlayer_ {
 public:
  using BasePlayer = BasePlayer_;
  using Spec = BasePlayer::Spec;
  using Game = BasePlayer::Game;
  using State = Game::State;
  using Move = Game::Move;
  using MoveSet = Game::MoveSet;
  using GameOutcome = Game::Types::GameOutcome;
  using TensorEncodings = Spec::EvalSpec::TensorEncodings;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using ActionRequest = core::ActionRequest<Game>;
  using ActionResponse = core::ActionResponse<Game>;
  using ChanceEventHandleRequest = core::ChanceEventHandleRequest<Game>;

  using TrainingInfo = Spec::TrainingInfo;

  using SearchResults = BasePlayer::SearchResults;
  using SearchResponse = BasePlayer::SearchResponse;

  using TrainingDataWriter = search::TrainingDataWriter<Spec>;
  using GameWriteLog = TrainingDataWriter::GameWriteLog;
  using GameWriteLog_sptr = TrainingDataWriter::GameWriteLog_sptr;

  template <typename... Ts>
  DataExportingPlayer(Ts&&... args)
      : BasePlayer(std::forward<Ts>(args)...), writer_(TrainingDataWriter::instance()) {}

  core::yield_instruction_t handle_chance_event(const ChanceEventHandleRequest&) override;
  bool start_game() override;
  void end_game(const State&, const GameOutcome&) override;

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
