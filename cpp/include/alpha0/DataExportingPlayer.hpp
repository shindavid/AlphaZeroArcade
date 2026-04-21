#pragma once

#include "alpha0/GameLog.hpp"
#include "alpha0/TrainingInfo.hpp"
#include "core/ActionRequest.hpp"
#include "core/ActionResponse.hpp"
#include "core/ChanceEventHandleRequest.hpp"
#include "search/TrainingDataWriter.hpp"

namespace alpha0 {

/*
 * A player that exports training data to a file via TrainingDataWriter.
 *
 * Assumes that BasePlayer is alpha0::Player<Spec>.
 */
template <typename BasePlayer_>
class DataExportingPlayer : public BasePlayer_ {
 public:
  using BasePlayer = BasePlayer_;
  using Spec = BasePlayer::Spec;
  using Game = BasePlayer::Game;
  using InfoSet = Game::InfoSet;
  using Move = Game::Move;
  using MoveSet = Game::MoveSet;
  using GameOutcome = Game::Types::GameOutcome;
  using TensorEncodings = Spec::TensorEncodings;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using ActionRequest = core::ActionRequest<Game>;
  using ActionResponse = core::ActionResponse<Game>;
  using ChanceEventHandleRequest = core::ChanceEventHandleRequest<Game>;

  using TrainingInfo = ::alpha0::TrainingInfo<Spec>;

  using SearchResults = BasePlayer::SearchResults;
  using SearchResponse = BasePlayer::SearchResponse;

  using GameWriteLog = ::alpha0::GameWriteLog<Spec>;
  using TrainingDataWriter = search::TrainingDataWriter<GameWriteLog>;
  using GameWriteLog_sptr = std::shared_ptr<GameWriteLog>;

  template <typename... Ts>
  DataExportingPlayer(Ts&&... args)
      : BasePlayer(std::forward<Ts>(args)...), writer_(TrainingDataWriter::instance()) {}

  core::yield_instruction_t handle_chance_event(const ChanceEventHandleRequest&) override;
  bool start_game() override;
  void end_game(const InfoSet&, const GameOutcome&) override;

 protected:
  ActionResponse get_action_response_helper(const SearchResults*, const ActionRequest&) override;

  void add_to_game_log(const ActionRequest&, const ActionResponse&, const SearchResults*);
  void extract_policy_target(const SearchResults* results);

  TrainingDataWriter* writer_;
  TrainingInfo training_info_;
  GameWriteLog_sptr game_log_;
  bool mid_handle_chance_event_ = false;
};

}  // namespace alpha0

#include "inline/alpha0/DataExportingPlayer.inl"
