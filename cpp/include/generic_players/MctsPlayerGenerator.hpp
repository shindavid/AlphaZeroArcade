#pragma once

#include <map>
#include <string>
#include <vector>

#include <generic_players/DataExportingMctsPlayer.hpp>
#include <generic_players/MctsPlayer.hpp>
#include <core/AbstractPlayerGenerator.hpp>
#include <core/concepts/Game.hpp>
#include <mcts/Constants.hpp>
#include <mcts/Manager.hpp>
#include <mcts/ManagerParams.hpp>

namespace generic {

template <core::concepts::Game Game>
class MctsPlayerGeneratorBase : public core::AbstractPlayerGenerator<Game> {
 public:
  using MctsManagerParams = mcts::ManagerParams<Game>;
  using MctsManager = mcts::Manager<Game>;
  using BaseMctsPlayer = generic::MctsPlayer<Game>;

  MctsPlayerGeneratorBase(mcts::Mode mode) : manager_params_(mode) {}

  /*
   * If this generator already generated a player for the given game_thread_id, dispatches to
   * generate_from_manager(), passing in the mcts::Manager* of that previous player. Otherwise,
   * dispatches to generate_from_scratch().
   */
  core::AbstractPlayer<Game>* generate(core::game_thread_id_t game_thread_id) override;

  void end_session() override;

 protected:
  virtual BaseMctsPlayer* generate_helper(MctsManager* manager, bool owns_manager) = 0;

  void validate_params();

  using manager_vec_t = std::vector<MctsManager*>;
  using manager_map_t = std::map<core::game_thread_id_t, manager_vec_t>;

  static manager_map_t manager_cache_;

  MctsManagerParams manager_params_;
};

template <core::concepts::Game Game>
class CompetitiveMctsPlayerGenerator : public MctsPlayerGeneratorBase<Game> {
 public:
  using base_t = MctsPlayerGeneratorBase<Game>;
  using BaseMctsPlayer = base_t::BaseMctsPlayer;
  using MctsManager = base_t::MctsManager;
  using MctsPlayer = generic::MctsPlayer<Game>;
  using MctsPlayerParams = MctsPlayer::Params;

  CompetitiveMctsPlayerGenerator();
  std::string get_default_name() const override;
  std::vector<std::string> get_types() const override { return {"MCTS-C", "MCTS-Competitive"}; }
  std::string get_description() const override { return "Competitive MCTS player"; }
  void print_help(std::ostream& s) override { make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;

 protected:
  auto make_options_description() {
    return this->manager_params_.make_options_description().add(
        mcts_player_params_.make_options_description());
  }

  BaseMctsPlayer* generate_helper(MctsManager* manager, bool owns_manager) override;

  MctsPlayerParams mcts_player_params_;
};

template <core::concepts::Game Game>
class TrainingMctsPlayerGenerator : public MctsPlayerGeneratorBase<Game> {
 public:
  using base_t = MctsPlayerGeneratorBase<Game>;
  using BaseMctsPlayer = base_t::BaseMctsPlayer;
  using MctsManager = base_t::MctsManager;
  using MctsPlayer = generic::DataExportingMctsPlayer<Game>;
  using MctsPlayerParams = MctsPlayer::Params;
  using TrainingDataWriterParams = MctsPlayer::TrainingDataWriterParams;

  TrainingMctsPlayerGenerator();
  std::string get_default_name() const override;
  std::vector<std::string> get_types() const override { return {"MCTS-T", "MCTS-Training"}; }
  std::string get_description() const override { return "Training MCTS player"; }
  void print_help(std::ostream& s) override { make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;

 protected:
  auto make_options_description() {
    return this->manager_params_.make_options_description()
        .add(mcts_player_params_.make_options_description())
        .add(writer_params_.make_options_description());
  }

  BaseMctsPlayer* generate_helper(MctsManager* manager, bool owns_manager) override;

  MctsPlayerParams mcts_player_params_;
  TrainingDataWriterParams writer_params_;
};

}  // namespace generic

#include <inline/generic_players/MctsPlayerGenerator.inl>
