#pragma once

#include "core/AbstractPlayerGenerator.hpp"
#include "core/BasicTypes.hpp"
#include "core/GameServerBase.hpp"
#include "core/PlayerFactory.hpp"
#include "generic_players/DataExportingPlayer.hpp"
#include "generic_players/alpha0/Player.hpp"
#include "search/Constants.hpp"
#include "search/VerboseManager.hpp"
#include "search/concepts/TraitsConcept.hpp"

#include <magic_enum/magic_enum_format.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace generic::alpha0 {

template <search::concepts::Traits Traits, typename PlayerT,
          search::Mode Mode = search::kCompetition>
class PlayerGeneratorBase : public core::AbstractPlayerGenerator<typename Traits::Game> {
 public:
  static constexpr int kDefaultMutexPoolSize = 1024;

  using Game = Traits::Game;
  using ManagerParams = Traits::ManagerParams;
  using BasePlayer = generic::alpha0::Player<Traits>;
  using PlayerParams = BasePlayer::Params;
  using SharedData = BasePlayer::SharedData;
  using SharedData_sptr = std::shared_ptr<SharedData>;

  using shared_data_vec_t = std::vector<SharedData_sptr>;
  using shared_data_map_t = std::map<core::game_slot_index_t, shared_data_vec_t>;

  static_assert(std::is_base_of_v<BasePlayer, PlayerT>,
                "PlayerT must be derived from generic::alpha0::Player<Traits>");

  PlayerGeneratorBase(core::GameServerBase*, shared_data_map_t& shared_data_cache);

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
  ManagerParams manager_params_;
  PlayerParams mcts_player_params_;
  shared_data_map_t& shared_data_cache_;
  core::mutex_vec_sptr_t common_node_mutex_pool_;     // only used in multi-threaded mode
  core::mutex_vec_sptr_t common_context_mutex_pool_;  // only used in multi-threaded mode
};

template <search::concepts::Traits Traits>
using CompetitionPlayerGenerator = PlayerGeneratorBase<Traits, generic::alpha0::Player<Traits>>;

template <search::concepts::Traits Traits>
using TrainingPlayerGeneratorBase =
  PlayerGeneratorBase<Traits, generic::DataExportingPlayer<generic::alpha0::Player<Traits>>,
                      search::kTraining>;

template <search::concepts::Traits Traits>
class TrainingPlayerGenerator : public TrainingPlayerGeneratorBase<Traits> {
 public:
  using Base = TrainingPlayerGeneratorBase<Traits>;
  using Game = Base::Game;
  using Base::Base;

  void end_session() override;
};

template <typename GeneratorT>
class Subfactory : public core::PlayerSubfactoryBase<typename GeneratorT::Game> {
 public:
  using shared_data_map_t = GeneratorT::shared_data_map_t;

  GeneratorT* create(core::GameServerBase* server) override {
    return new GeneratorT(server, shared_data_cache_);
  }

 private:
  shared_data_map_t shared_data_cache_;
};

}  // namespace generic::alpha0

namespace core {

template <search::concepts::Traits Traits>
class PlayerSubfactory<generic::alpha0::CompetitionPlayerGenerator<Traits>>
    : public generic::alpha0::Subfactory<generic::alpha0::CompetitionPlayerGenerator<Traits>> {};

template <search::concepts::Traits Traits>
class PlayerSubfactory<generic::alpha0::TrainingPlayerGenerator<Traits>>
    : public generic::alpha0::Subfactory<generic::alpha0::TrainingPlayerGenerator<Traits>> {};

}  // namespace core

#include "inline/generic_players/alpha0/PlayerGenerator.inl"
