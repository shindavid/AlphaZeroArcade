#pragma once

#include <generic_players/MctsPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/GameLog.hpp>
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

  using GameLogWriter = core::GameLogWriter<Game>;

  using State = Game::State;
  using ActionMask = Game::Types::ActionMask;
  using ValueTensor = Game::Types::ValueTensor;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;

  using TrainingDataWriter = core::TrainingDataWriter<Game>;
  using TrainingDataWriterParams = TrainingDataWriter::Params;

  using base_t = MctsPlayer<Game>;
  using Params = base_t::Params;
  using MctsManager = base_t::MctsManager;
  using SearchResults = base_t::SearchResults;

  template <typename... BaseArgs>
  DataExportingMctsPlayer(const TrainingDataWriterParams& writer_params, BaseArgs&&...);

  void start_game() override;
  void receive_state_change(core::seat_index_t seat, const State& state,
                            core::action_t action) override;
  core::ActionResponse get_action_response(const State&, const ActionMask&) override;
  void end_game(const State&, const ValueTensor&) override;

 protected:
  static void extract_policy_target(const SearchResults* results, PolicyTensor** target);

  TrainingDataWriter* writer_;
  TrainingDataWriter::GameLogWriter_sptr game_log_ = nullptr;
};

}  // namespace generic

#include <inline/generic_players/DataExportingMctsPlayer.inl>
