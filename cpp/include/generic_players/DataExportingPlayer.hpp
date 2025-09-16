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
  using GameWriteLog_sptr = core::TrainingDataWriter<Game>::GameWriteLog_sptr;
  using State = Game::State;
  using ActionMask = Game::Types::ActionMask;
  using ValueTensor = Game::Types::ValueTensor;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using ActionRequest = Game::Types::ActionRequest;
  using ActionResponse = Game::Types::ActionResponse;
  using ChangeEventPreHandleRequest = Game::Types::ChangeEventPreHandleRequest;
  using ChanceEventPreHandleResponse = Game::Types::ChanceEventPreHandleResponse;
  using TrainingInfo = Game::Types::TrainingInfo;

  using SearchResults = BasePlayer::SearchResults;
  using SearchResponse = BasePlayer::SearchResponse;

  using BasePlayer::BasePlayer;

  ChanceEventPreHandleResponse prehandle_chance_event(const ChangeEventPreHandleRequest&) override;

 protected:
  virtual ActionResponse get_action_response_helper(const SearchResults*,
                                                    const ActionMask& valid_actions) override;
  static void extract_policy_target(const SearchResults* results, PolicyTensor** target);

  PolicyTensor policy_target_;
  ActionValueTensor action_values_target_;

  bool use_for_training_;
  bool previous_used_for_training_;
  bool mid_prehandle_chance_event_ = false;
};

}  // namespace generic

#include "inline/generic_players/DataExportingPlayer.inl"
