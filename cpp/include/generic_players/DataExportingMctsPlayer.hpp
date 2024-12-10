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

  using GameLogWriter_sptr = core::TrainingDataWriter<Game>::GameLogWriter_sptr;
  using State = Game::State;
  using ActionMask = Game::Types::ActionMask;
  using ValueTensor = Game::Types::ValueTensor;
  using Policy = Game::Types::Policy;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using ActionResponse = Game::Types::ActionResponse;
  using TrainingInfo = Game::Types::TrainingInfo;

  using base_t = MctsPlayer<Game>;
  using Params = base_t::Params;
  using MctsManager = base_t::MctsManager;
  using SearchResults = base_t::SearchResults;

  using base_t::base_t;

  ActionResponse get_action_response(const State&, const ActionMask&) override;

 protected:
  static void extract_policy_target(const SearchResults* results, Policy** target);

  Policy policy_target_;
  ActionValueTensor action_values_target_;
};

}  // namespace generic

#include <inline/generic_players/DataExportingMctsPlayer.inl>
