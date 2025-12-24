#pragma once

#include "core/ActionRequest.hpp"
#include "core/ActionResponse.hpp"
#include "core/ChanceEventHandleRequest.hpp"
#include "search/AlgorithmsFor.hpp"
#include "search/TrainingDataWriter.hpp"
#include "search/TrainingInfoParams.hpp"

namespace generic {

/*
 * A generic player that exports training data to a file via TrainingDataWriter.
 *
 * Assumes that BasePlayer is either one of the following:
 *
 * - generic::alpha0::Player<Traits>
 * - generic::beta0::Player<Traits>
 */
template <typename BasePlayer_>
class DataExportingPlayer : public BasePlayer_ {
 public:
  using BasePlayer = BasePlayer_;
  using Traits = BasePlayer::Traits;
  using Game = BasePlayer::Game;
  using State = Game::State;
  using ActionMask = Game::Types::ActionMask;
  using GameResultTensor = Game::Types::GameResultTensor;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using ActionRequest = core::ActionRequest<Game>;
  using ChanceEventHandleRequest = core::ChanceEventHandleRequest<Game>;

  using TrainingInfo = Traits::TrainingInfo;
  using Algorithms = search::AlgorithmsForT<Traits>;

  using SearchResults = BasePlayer::SearchResults;
  using SearchResponse = BasePlayer::SearchResponse;

  using TrainingInfoParams = search::TrainingInfoParams<Traits>;
  using TrainingDataWriter = search::TrainingDataWriter<Traits>;
  using GameWriteLog = TrainingDataWriter::GameWriteLog;
  using GameWriteLog_sptr = TrainingDataWriter::GameWriteLog_sptr;

  template <typename... Ts>
  DataExportingPlayer(Ts&&... args)
      : BasePlayer(std::forward<Ts>(args)...), writer_(TrainingDataWriter::instance()) {}

  core::yield_instruction_t handle_chance_event(const ChanceEventHandleRequest&) override;
  bool start_game() override;
  void end_game(const State&, const GameResultTensor&) override;

 protected:
  core::ActionResponse get_action_response_helper(const SearchResults*,
                                                  const ActionRequest&) override;

  void add_to_game_log(const ActionRequest&, const core::ActionResponse&, const SearchResults*);
  void extract_policy_target(const SearchResults* results);

  TrainingDataWriter* const writer_;
  TrainingInfo training_info_;
  GameWriteLog_sptr game_log_;
  bool mid_handle_chance_event_ = false;
};

}  // namespace generic

#include "inline/generic_players/DataExportingPlayer.inl"
