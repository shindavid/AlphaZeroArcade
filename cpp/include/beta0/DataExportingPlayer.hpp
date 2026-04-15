#pragma once

#include "beta0/GameLog.hpp"
#include "beta0/TrainingInfo.hpp"
#include "core/ActionRequest.hpp"
#include "core/ActionResponse.hpp"
#include "core/ChanceEventHandleRequest.hpp"
#include "search/TrainingDataWriter.hpp"

namespace beta0 {

/*
 * A player that exports training data to a file via TrainingDataWriter.
 *
 * Assumes that BasePlayer is beta0::Player<Spec>.
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
  using TensorEncodings = Spec::TensorEncodings;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using ActionRequest = core::ActionRequest<Game>;
  using ActionResponse = core::ActionResponse<Game>;
  using ChanceEventHandleRequest = core::ChanceEventHandleRequest<Game>;

  using TrainingInfo = ::beta0::TrainingInfo<Spec>;

  using SearchResults = BasePlayer::SearchResults;
  using SearchResponse = BasePlayer::SearchResponse;

  using GameWriteLog = ::beta0::GameWriteLog<Spec>;
  using TrainingDataWriter = search::TrainingDataWriter<GameWriteLog>;
  using GameWriteLog_sptr = std::shared_ptr<GameWriteLog>;

  template <typename... Ts>
  DataExportingPlayer(Ts&&... args)
      : BasePlayer(std::forward<Ts>(args)...), writer_(TrainingDataWriter::instance()) {}

  bool start_game() override;
  void end_game(const State&, const GameOutcome&) override;

 protected:
  TrainingDataWriter* writer_;
  TrainingInfo training_info_;
  GameWriteLog_sptr game_log_;
};

}  // namespace beta0
