#pragma once

#include <generic_players/MctsPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <core/TrainingDataWriter.hpp>

#include <vector>

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
  using TrainingInfo = Game::Types::TrainingInfo;

  using base_t = MctsPlayer<Game>;
  using Params = base_t::Params;
  using MctsManager = base_t::MctsManager;
  using SearchResults = base_t::SearchResults;

  using base_t::base_t;

  ActionResponse get_action_response(const ActionRequest& request) override;
  ActionValueTensor* prehandle_chance_event() override;

 protected:
  static void extract_policy_target(const SearchResults* results, PolicyTensor** target);

  PolicyTensor policy_target_;
  ActionValueTensor action_values_target_;

  bool use_for_training_;
  bool previous_used_for_training_;
};

}  // namespace generic

#include <inline/generic_players/DataExportingMctsPlayer.inl>
