#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <core/TrainingDataWriter.hpp>
#include <generic_players/MctsPlayer.hpp>

namespace generic {

/*
 * A variant of MctsPlayer that exports training data to a file via TrainingDataWriter.
 */
template <core::concepts::Game Game>
class DataExportingMctsPlayer : public MctsPlayer<Game> {
 public:
  /*
   * The argument for using a full search is so that the opp reply target is more accurate.
   *
   * The argument against is that the opp reply target is not that important, making full searches
   * for that purpose an inefficient use of compute budget.
   */
  static constexpr bool kForceFullSearchIfRecordingAsOppReply = false;

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

  using base_t = MctsPlayer<Game>;
  using Params = base_t::Params;
  using MctsManager = base_t::MctsManager;
  using SearchResults = base_t::SearchResults;
  using SearchRequest = base_t::SearchRequest;
  using SearchResponse = base_t::SearchResponse;

  using base_t::base_t;

  ActionResponse get_action_response(const ActionRequest& request) override;
  ChanceEventPreHandleResponse prehandle_chance_event(const ChangeEventPreHandleRequest&) override;

 protected:
  static void extract_policy_target(const SearchResults* results, PolicyTensor** target);

  PolicyTensor policy_target_;
  ActionValueTensor action_values_target_;

  bool use_for_training_;
  bool previous_used_for_training_;
  bool mid_prehandle_chance_event_ = false;
};

}  // namespace generic

#include <inline/generic_players/DataExportingMctsPlayer.inl>
