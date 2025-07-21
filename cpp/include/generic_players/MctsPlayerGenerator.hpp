#pragma once

#include "core/AbstractPlayerGenerator.hpp"
#include "core/GameServerBase.hpp"
#include "core/PlayerFactory.hpp"
#include "core/concepts/Game.hpp"
#include "generic_players/DataExportingMctsPlayer.hpp"
#include "generic_players/MctsPlayer.hpp"
#include "mcts/Constants.hpp"
#include "mcts/Manager.hpp"
#include "mcts/ManagerParams.hpp"
#include "mcts/TypeDefs.hpp"

#include <magic_enum/magic_enum_format.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace generic {

template <core::concepts::Game Game, typename PlayerT, mcts::Mode Mode = mcts::kCompetitive>
class MctsPlayerGeneratorBase : public core::AbstractPlayerGenerator<Game> {
 public:
  static constexpr int kDefaultMutexPoolSize = 1024;

  using MctsManagerParams = mcts::ManagerParams<Game>;
  using MctsManager = mcts::Manager<Game>;
  using BaseMctsPlayer = generic::MctsPlayer<Game>;
  using MctsPlayerParams = BaseMctsPlayer::Params;
  using SharedData = BaseMctsPlayer::SharedData;
  using SharedData_sptr = std::shared_ptr<SharedData>;

  using shared_data_vec_t = std::vector<SharedData_sptr>;
  using shared_data_map_t = std::map<core::game_slot_index_t, shared_data_vec_t>;

  static_assert(std::is_base_of_v<BaseMctsPlayer, PlayerT>,
                "PlayerT must be derived from generic::MctsPlayer<Game>");

  MctsPlayerGeneratorBase(core::GameServerBase*, shared_data_map_t& shared_data_cache);

  /*
   * If this generator already generated a player for the given game_slot_index_t, dispatches to
   * generate_from_manager(), passing in the mcts::Manager* of that previous player. Otherwise,
   * dispatches to generate_from_scratch().
   */
  core::AbstractPlayer<Game>* generate(core::game_slot_index_t game_slot_index) override;

  void end_session() override;
  std::string get_default_name() const override;
  void print_help(std::ostream& s) override { make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;
  std::vector<std::string> get_types() const override;
  std::string get_description() const override;

 protected:
  auto make_options_description() {
    return manager_params_.make_options_description().add(
      mcts_player_params_.make_options_description());
  }

  PlayerT* generate_helper(SharedData_sptr& shared_data, bool owns_shared_data);
  SharedData_sptr generate_shared_data();

  void validate_params();

  core::GameServerBase* const server_;
  MctsManagerParams manager_params_;
  MctsPlayerParams mcts_player_params_;
  shared_data_map_t& shared_data_cache_;
  mcts::mutex_vec_sptr_t common_node_mutex_pool_;     // only used in multi-threaded mode
  mcts::mutex_vec_sptr_t common_context_mutex_pool_;  // only used in multi-threaded mode
};

template <core::concepts::Game Game>
using CompetitiveMctsPlayerGenerator = MctsPlayerGeneratorBase<Game, generic::MctsPlayer<Game>>;

template <core::concepts::Game Game>
using TrainingMctsPlayerGenerator =
  MctsPlayerGeneratorBase<Game, generic::DataExportingMctsPlayer<Game>, mcts::kTraining>;

template <typename GeneratorT>
class MctsSubfactory : public core::PlayerSubfactoryBase<typename GeneratorT::Game> {
 public:
  using shared_data_map_t = GeneratorT::shared_data_map_t;

  GeneratorT* create(core::GameServerBase* server) override {
    return new GeneratorT(server, shared_data_cache_);
  }

 private:
  shared_data_map_t shared_data_cache_;
};

}  // namespace generic

namespace core {

template <core::concepts::Game Game>
class PlayerSubfactory<generic::CompetitiveMctsPlayerGenerator<Game>>
    : public generic::MctsSubfactory<generic::CompetitiveMctsPlayerGenerator<Game>> {};

template <core::concepts::Game Game>
class PlayerSubfactory<generic::TrainingMctsPlayerGenerator<Game>>
    : public generic::MctsSubfactory<generic::TrainingMctsPlayerGenerator<Game>> {};

}  // namespace core

#include "inline/generic_players/MctsPlayerGenerator.inl"
