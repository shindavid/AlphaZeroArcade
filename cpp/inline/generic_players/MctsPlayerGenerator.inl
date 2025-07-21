#include "generic_players/MctsPlayerGenerator.hpp"

#include "core/LoopControllerClient.hpp"
#include "mcts/Constants.hpp"

#include <format>

namespace generic {

// MctsPlayerGeneratorBase

template <core::concepts::Game Game, typename PlayerT, mcts::Mode Mode>
MctsPlayerGeneratorBase<Game, PlayerT, Mode>::MctsPlayerGeneratorBase(
  core::GameServerBase* server, shared_data_map_t& shared_data_cache)
    : server_(server),
      manager_params_(Mode),
      mcts_player_params_(Mode),
      shared_data_cache_(shared_data_cache) {}

template <core::concepts::Game Game, typename PlayerT, mcts::Mode Mode>
core::AbstractPlayer<Game>* MctsPlayerGeneratorBase<Game, PlayerT, Mode>::generate(
  core::game_slot_index_t game_slot_index) {
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

template <core::concepts::Game Game, typename PlayerT, mcts::Mode Mode>
void MctsPlayerGeneratorBase<Game, PlayerT, Mode>::end_session() {
  for (auto& pair : shared_data_cache_) {
    for (SharedData_sptr& shared_data : pair.second) {
      shared_data->manager.end_session();
    }
  }
}

template <core::concepts::Game Game, typename PlayerT, mcts::Mode Mode>
std::string MctsPlayerGeneratorBase<Game, PlayerT, Mode>::get_default_name() const {
  return std::format("{}-{}", this->get_types()[0], mcts_player_params_.num_fast_iters);
}

template <core::concepts::Game Game, typename PlayerT, mcts::Mode Mode>
void MctsPlayerGeneratorBase<Game, PlayerT, Mode>::parse_args(
  const std::vector<std::string>& args) {
  this->parse_args_helper(make_options_description(), args);
}

template <core::concepts::Game Game, typename PlayerT, mcts::Mode Mode>
std::vector<std::string> MctsPlayerGeneratorBase<Game, PlayerT, Mode>::get_types() const {
  if (Mode == mcts::kCompetitive) {
    return {"MCTS-C", "MCTS-Competitive"};
  } else if (Mode == mcts::kTraining) {
    return {"MCTS-T", "MCTS-Training"};
  } else {
    throw util::CleanException("Unknown mcts::Mode: {}", Mode);
  }
}

template <core::concepts::Game Game, typename PlayerT, mcts::Mode Mode>
std::string MctsPlayerGeneratorBase<Game, PlayerT, Mode>::get_description() const {
  if (Mode == mcts::kCompetitive) {
    return "Competitive MCTS player";
  } else if (Mode == mcts::kTraining) {
    return "Training MCTS player";
  } else {
    throw util::CleanException("Unknown mcts::Mode: {}", Mode);
  }
}

template <core::concepts::Game Game, typename PlayerT, mcts::Mode Mode>
PlayerT* MctsPlayerGeneratorBase<Game, PlayerT, Mode>::generate_helper(SharedData_sptr& shared_data,
                                                                       bool owns_shared_data) {
  return new PlayerT(this->mcts_player_params_, shared_data, owns_shared_data);
}

template <core::concepts::Game Game, typename PlayerT, mcts::Mode Mode>
typename MctsPlayerGeneratorBase<Game, PlayerT, Mode>::SharedData_sptr
MctsPlayerGeneratorBase<Game, PlayerT, Mode>::generate_shared_data() {
  if (manager_params_.num_search_threads == 1) {
    return std::make_shared<SharedData>(manager_params_, server_);
  } else {
    if (!common_node_mutex_pool_.get()) {
      common_node_mutex_pool_ = std::make_shared<mcts::mutex_vec_t>(kDefaultMutexPoolSize);
    }
    if (!common_context_mutex_pool_.get()) {
      common_context_mutex_pool_ = std::make_shared<mcts::mutex_vec_t>(kDefaultMutexPoolSize);
    }
    return std::make_shared<SharedData>(common_node_mutex_pool_, common_context_mutex_pool_,
                                        manager_params_, server_);
  }
}

}  // namespace generic
