#pragma once

#include <core/AbstractPlayerGenerator.hpp>
#include <core/concepts/Game.hpp>
#include <generic_players/DataExportingMctsPlayer.hpp>
#include <generic_players/MctsPlayer.hpp>
#include <mcts/Constants.hpp>
#include <mcts/Manager.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/TypeDefs.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace generic {

template <core::concepts::Game Game>
class MctsPlayerGeneratorBase : public core::AbstractPlayerGenerator<Game> {
 public:
  static constexpr int kDefaultMutexPoolSize = 1024;

  using MctsManagerParams = mcts::ManagerParams<Game>;
  using MctsManager = mcts::Manager<Game>;
  using BaseMctsPlayer = generic::MctsPlayer<Game>;
  using SharedData = BaseMctsPlayer::SharedData;
  using SharedData_sptr = std::shared_ptr<SharedData>;

  MctsPlayerGeneratorBase(mcts::Mode mode) : manager_params_(mode) {}

  /*
   * If this generator already generated a player for the given game_slot_index_t, dispatches to
   * generate_from_manager(), passing in the mcts::Manager* of that previous player. Otherwise,
   * dispatches to generate_from_scratch().
   */
  core::AbstractPlayer<Game>* generate(core::game_slot_index_t game_slot_index) override;

  void end_session() override;

 protected:
  virtual BaseMctsPlayer* generate_helper(SharedData_sptr& shared_data, bool owns_shared_data) = 0;
  SharedData_sptr generate_shared_data();

  void validate_params();

  using shared_data_vec_t = std::vector<SharedData_sptr>;
  using shared_data_map_t = std::map<core::game_slot_index_t, shared_data_vec_t>;

  // TODO: instead of making this static, make this an object passed in via a specialized
  // PlayerSubfactory's create() method.
  static shared_data_map_t shared_data_cache_;

  MctsManagerParams manager_params_;
  mcts::mutex_vec_sptr_t common_mutex_pool_;  // only used in multi-threaded mode
};

template <core::concepts::Game Game>
class CompetitiveMctsPlayerGenerator : public MctsPlayerGeneratorBase<Game> {
 public:
  using base_t = MctsPlayerGeneratorBase<Game>;
  using BaseMctsPlayer = base_t::BaseMctsPlayer;
  using MctsManager = base_t::MctsManager;
  using MctsPlayer = generic::MctsPlayer<Game>;
  using MctsPlayerParams = MctsPlayer::Params;
  using SharedData = BaseMctsPlayer::SharedData;
  using SharedData_sptr = std::shared_ptr<SharedData>;

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

  BaseMctsPlayer* generate_helper(SharedData_sptr& shared_data, bool owns_shared_data) override;

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
  using SharedData = BaseMctsPlayer::SharedData;
  using SharedData_sptr = std::shared_ptr<SharedData>;

  TrainingMctsPlayerGenerator();
  std::string get_default_name() const override;
  std::vector<std::string> get_types() const override { return {"MCTS-T", "MCTS-Training"}; }
  std::string get_description() const override { return "Training MCTS player"; }
  void print_help(std::ostream& s) override { make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;

 protected:
  auto make_options_description() {
    return this->manager_params_.make_options_description()
        .add(mcts_player_params_.make_options_description());
  }

  BaseMctsPlayer* generate_helper(SharedData_sptr& shared_data, bool owns_shared_data) override;

  MctsPlayerParams mcts_player_params_;
};

}  // namespace generic

#include <inline/generic_players/MctsPlayerGenerator.inl>
