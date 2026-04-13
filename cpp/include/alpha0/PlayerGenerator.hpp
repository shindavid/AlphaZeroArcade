#pragma once

#include "alpha0/ManagerParams.hpp"
#include "core/AbstractPlayerGenerator.hpp"
#include "core/BasicTypes.hpp"
#include "core/GameServerBase.hpp"
#include "core/PlayerFactory.hpp"
#include "core/SearchParadigm.hpp"
#include "generic_players/DataExportingPlayer.hpp"
#include "search/Constants.hpp"

#include <magic_enum/magic_enum_format.hpp>

#include <format>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace alpha0 {

template <typename PlayerT, search::Mode Mode>
class PlayerGeneratorBase : public core::AbstractPlayerGenerator<typename PlayerT::Spec::Game> {
 public:
  static constexpr int kDefaultMutexPoolSize = 1024;

  using Spec = PlayerT::Spec;
  using Game = Spec::Game;
  using ManagerParams = alpha0::ManagerParams<Spec>;
  using BasePlayer = PlayerT::BasePlayer;
  using PlayerParams = BasePlayer::Params;
  using SharedData = BasePlayer::SharedData;
  using SharedData_sptr = std::shared_ptr<SharedData>;

  using shared_data_vec_t = std::vector<SharedData_sptr>;
  using shared_data_map_t = std::map<core::game_slot_index_t, shared_data_vec_t>;

  static_assert(std::is_base_of_v<BasePlayer, PlayerT>);

  PlayerGeneratorBase(core::GameServerBase*, shared_data_map_t& shared_data_cache);

  /*
   * If this generator already generated a player for the given game_slot_index_t, dispatches to
   * generate_from_manager(), passing in the Manager* of that previous player. Otherwise,
   * dispatches to generate_from_scratch().
   */
  core::AbstractPlayer<Game>* generate(core::game_slot_index_t game_slot_index) override;

  void end_session() override;
  std::string get_default_name() const override;
  void print_help(std::ostream& s) override { make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;

  std::string type_str() const override {
    constexpr auto kParadigm = Spec::kParadigm;
    if constexpr (Mode == search::kCompetition) {
      return std::format("{}-C", core::SearchParadigmTraits<kParadigm>::kName);
    } else {
      return std::format("{}-T", core::SearchParadigmTraits<kParadigm>::kName);
    }
  }

  std::string get_description() const override {
    constexpr auto kParadigm = Spec::kParadigm;
    if constexpr (Mode == search::kCompetition) {
      return std::format("Competition {} player", core::SearchParadigmTraits<kParadigm>::kName);
    } else {
      return std::format("Training {} player", core::SearchParadigmTraits<kParadigm>::kName);
    }
  }

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

template <typename PlayerT>
using CompetitionPlayerGenerator = PlayerGeneratorBase<PlayerT, search::kCompetition>;

template <typename PlayerT>
using TrainingPlayerGeneratorBase =
  PlayerGeneratorBase<generic::DataExportingPlayer<PlayerT>, search::kTraining>;

template <typename PlayerT>
class TrainingPlayerGenerator : public TrainingPlayerGeneratorBase<PlayerT> {
 public:
  using Base = TrainingPlayerGeneratorBase<PlayerT>;
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

}  // namespace alpha0

#include "inline/alpha0/PlayerGenerator.inl"
