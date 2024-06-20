#pragma once

#include <generic_players/MctsPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/GameLog.hpp>
#include <core/concepts/Game.hpp>
#include <core/TrainingDataWriter.hpp>
#include <core/SearchResults.hpp>

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

  using GameLogWriter = core::GameLogWriter<Game>;

  using FullState = typename Game::FullState;
  using ActionMask = typename Game::Types::ActionMask;
  using ValueArray = typename Game::Types::ValueArray;
  using PolicyTensor = typename Game::Types::PolicyTensor;

  using TrainingDataWriter = core::TrainingDataWriter<Game>;
  using TrainingDataWriterParams = typename TrainingDataWriter::Params;

  using base_t = MctsPlayer<Game>;
  using Params = base_t::Params;
  using MctsManager = base_t::MctsManager;
  using SearchResults = base_t::SearchResults;

  template <typename... BaseArgs>
  DataExportingMctsPlayer(const TrainingDataWriterParams& writer_params, BaseArgs&&...);

  void start_game() override;
  void receive_state_change(core::seat_index_t seat, const FullState& state,
                            core::action_t action) override;
  core::ActionResponse get_action_response(const FullState&, const ActionMask&) override;
  void end_game(const FullState&, const ValueArray&) override;

 protected:
  static void extract_policy_target(const SearchResults* results, PolicyTensor** target);

  TrainingDataWriter* writer_;
  TrainingDataWriter::GameLogWriter_sptr game_log_;
};

}  // namespace generic

#include <inline/generic_players/DataExportingMctsPlayer.inl>
