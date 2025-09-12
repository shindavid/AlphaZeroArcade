#pragma once

#include "alphazero/ManagerParams.hpp"
#include "core/AbstractPlayerGenerator.hpp"
#include "core/BasicTypes.hpp"
#include "core/GameServerBase.hpp"
#include "core/PlayerFactory.hpp"
#include "generic_players/DataExportingMctsPlayer.hpp"
#include "generic_players/MctsPlayer.hpp"
#include "search/Constants.hpp"
#include "search/Manager.hpp"
#include "search/concepts/TraitsConcept.hpp"

#include <magic_enum/magic_enum_format.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace generic {

template <search::concepts::Traits Traits, typename PlayerT,
          search::Mode Mode = search::kCompetitive>
class MctsPlayerGeneratorBase : public core::AbstractPlayerGenerator<typename Traits::Game> {
 public:
  static constexpr int kDefaultMutexPoolSize = 1024;

  using EvalSpec = Traits::EvalSpec;
  using Game = Traits::Game;
  using MctsManagerParams = alpha0::ManagerParams<EvalSpec>;
  using MctsManager = search::Manager<Traits>;
  using BaseMctsPlayer = generic::MctsPlayer<Traits>;
  using MctsPlayerParams = BaseMctsPlayer::Params;
  using SharedData = BaseMctsPlayer::SharedData;
  using SharedData_sptr = std::shared_ptr<SharedData>;

  using shared_data_vec_t = std::vector<SharedData_sptr>;
  using shared_data_map_t = std::map<core::game_slot_index_t, shared_data_vec_t>;

  static_assert(std::is_base_of_v<BaseMctsPlayer, PlayerT>,
                "PlayerT must be derived from generic::MctsPlayer<EvalSpec>");

  MctsPlayerGeneratorBase(core::GameServerBase*, shared_data_map_t& shared_data_cache);

  /*
   * If this generator already generated a player for the given game_slot_index_t, dispatches to
   * generate_from_manager(), passing in the search::Manager* of that previous player. Otherwise,
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
  core::mutex_vec_sptr_t common_node_mutex_pool_;     // only used in multi-threaded mode
  core::mutex_vec_sptr_t common_context_mutex_pool_;  // only used in multi-threaded mode
};

template <search::concepts::Traits Traits>
using CompetitiveMctsPlayerGenerator = MctsPlayerGeneratorBase<Traits, generic::MctsPlayer<Traits>>;

template <search::concepts::Traits Traits>
using TrainingMctsPlayerGenerator =
  MctsPlayerGeneratorBase<Traits, generic::DataExportingMctsPlayer<Traits>, search::kTraining>;

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

template <search::concepts::Traits Traits>
class PlayerSubfactory<generic::CompetitiveMctsPlayerGenerator<Traits>>
    : public generic::MctsSubfactory<generic::CompetitiveMctsPlayerGenerator<Traits>> {};

template <search::concepts::Traits Traits>
class PlayerSubfactory<generic::TrainingMctsPlayerGenerator<Traits>>
    : public generic::MctsSubfactory<generic::TrainingMctsPlayerGenerator<Traits>> {};

}  // namespace core

#include "inline/generic_players/MctsPlayerGenerator.inl"
