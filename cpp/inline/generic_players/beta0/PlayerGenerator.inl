#include "generic_players/beta0/PlayerGenerator.hpp"

#include "search/Constants.hpp"

#include <format>

namespace generic::beta0 {

// PlayerGeneratorBase

template <search::concepts::Traits Traits, typename PlayerT, search::Mode Mode>
PlayerGeneratorBase<Traits, PlayerT, Mode>::PlayerGeneratorBase(
  core::GameServerBase* server, shared_data_map_t& shared_data_cache)
    : server_(server),
      manager_params_(Mode),
      mcts_player_params_(Mode),
      shared_data_cache_(shared_data_cache) {}

template <search::concepts::Traits Traits, typename PlayerT, search::Mode Mode>
core::AbstractPlayer<typename Traits::Game>*
PlayerGeneratorBase<Traits, PlayerT, Mode>::generate(core::game_slot_index_t game_slot_index) {
  shared_data_vec_t& vec = shared_data_cache_[game_slot_index];
  for (SharedData_sptr& shared_data : vec) {
    if (shared_data->manager.params() == manager_params_) {
      return generate_helper(shared_data, false);
    }
  }

  SharedData_sptr shared_data = generate_shared_data();
  vec.push_back(shared_data);
  return generate_helper(shared_data, true);
}

template <search::concepts::Traits Traits, typename PlayerT, search::Mode Mode>
void PlayerGeneratorBase<Traits, PlayerT, Mode>::end_session() {
  for (auto& pair : shared_data_cache_) {
    for (SharedData_sptr& shared_data : pair.second) {
      shared_data->manager.end_session();
    }
  }
}

template <search::concepts::Traits Traits, typename PlayerT, search::Mode Mode>
std::string PlayerGeneratorBase<Traits, PlayerT, Mode>::get_default_name() const {
  return std::format("{}-{}", this->get_types()[0], mcts_player_params_.num_fast_iters);
}

template <search::concepts::Traits Traits, typename PlayerT, search::Mode Mode>
void PlayerGeneratorBase<Traits, PlayerT, Mode>::parse_args(
  const std::vector<std::string>& args) {
  this->parse_args_helper(make_options_description(), args);
}

template <search::concepts::Traits Traits, typename PlayerT, search::Mode Mode>
std::vector<std::string> PlayerGeneratorBase<Traits, PlayerT, Mode>::get_types() const {
  if (Mode == search::kCompetition) {
    return {"beta0-C", "BetaZero-Competition"};
  } else if (Mode == search::kTraining) {
    return {"beta0-T", "BetaZero-Training"};
  } else {
    throw util::CleanException("Unknown search::Mode: {}", Mode);
  }
}

template <search::concepts::Traits Traits, typename PlayerT, search::Mode Mode>
std::string PlayerGeneratorBase<Traits, PlayerT, Mode>::get_description() const {
  if (Mode == search::kCompetition) {
    return "Competition BetaZero player";
  } else if (Mode == search::kTraining) {
    return "Training BetaZero player";
  } else {
    throw util::CleanException("Unknown search::Mode: {}", Mode);
  }
}

template <search::concepts::Traits Traits, typename PlayerT, search::Mode Mode>
PlayerT* PlayerGeneratorBase<Traits, PlayerT, Mode>::generate_helper(
  SharedData_sptr& shared_data, bool owns_shared_data) {
  return new PlayerT(this->mcts_player_params_, shared_data, owns_shared_data);
}

template <search::concepts::Traits Traits, typename PlayerT, search::Mode Mode>
typename PlayerGeneratorBase<Traits, PlayerT, Mode>::SharedData_sptr
PlayerGeneratorBase<Traits, PlayerT, Mode>::generate_shared_data() {
  if (manager_params_.num_search_threads == 1) {
    return std::make_shared<SharedData>(manager_params_, server_);
  } else {
    if (!common_node_mutex_pool_.get()) {
      common_node_mutex_pool_ = std::make_shared<core::mutex_vec_t>(kDefaultMutexPoolSize);
    }
    if (!common_context_mutex_pool_.get()) {
      common_context_mutex_pool_ = std::make_shared<core::mutex_vec_t>(kDefaultMutexPoolSize);
    }
    return std::make_shared<SharedData>(common_node_mutex_pool_, common_context_mutex_pool_,
                                        manager_params_, server_);
  }
}

template <search::concepts::Traits Traits>
void TrainingPlayerGenerator<Traits>::end_session() {
  Base::end_session();

  using TrainingDataWriter = core::TrainingDataWriter<Game>;

  TrainingDataWriter* writer = TrainingDataWriter::instance();
  if (writer) {
    writer->wait_until_batch_empty();
  }
}

}  // namespace generic::beta0
