#include <generic_players/MctsPlayerGenerator.hpp>

#include <mcts/Constants.hpp>

namespace generic {

// MctsPlayerGeneratorBase

template <core::concepts::Game Game>
typename MctsPlayerGeneratorBase<Game>::shared_data_map_t
    MctsPlayerGeneratorBase<Game>::shared_data_cache_;

template <core::concepts::Game Game>
core::AbstractPlayer<Game>* MctsPlayerGeneratorBase<Game>::generate(
  core::game_slot_index_t game_slot_index) {
  shared_data_vec_t& vec = shared_data_cache_[game_slot_index];
  for (SharedData_sptr& shared_data : vec) {
    if (shared_data->manager.params() == manager_params_) {
      return generate_helper(shared_data, false);
    }
  }

  SharedData_sptr shared_data = std::make_shared<SharedData>(manager_params_);
  vec.push_back(shared_data);
  return generate_helper(shared_data, true);
}

template <core::concepts::Game Game>
void MctsPlayerGeneratorBase<Game>::end_session() {
  for (auto& pair : shared_data_cache_) {
    for (SharedData_sptr& shared_data : pair.second) {
      shared_data->manager.end_session(num_game_threads_);
    }
  }
}

template <core::concepts::Game Game>
typename MctsPlayerGeneratorBase<Game>::SharedData_sptr
MctsPlayerGeneratorBase<Game>::generate_shared_data() {
  if (manager_params_.num_search_threads == 1) {
    return std::make_shared<SharedData>(manager_params_);
  } else {
    if (!common_mutex_cv_pool_.get()) {
      common_mutex_cv_pool_ = std::make_shared<mcts::mutex_cv_vec_t>(kDefaultMutexPoolSize);
    }
    return std::make_shared<SharedData>(common_mutex_cv_pool_, manager_params_);
  }
}


// CompetitiveMctsPlayerGenerator

template <core::concepts::Game Game>
CompetitiveMctsPlayerGenerator<Game>::CompetitiveMctsPlayerGenerator()
    : base_t(mcts::kCompetitive), mcts_player_params_(mcts::kCompetitive) {}

template <core::concepts::Game Game>
std::string CompetitiveMctsPlayerGenerator<Game>::get_default_name() const {
  return util::create_string("MCTS-C-%d", mcts_player_params_.num_fast_iters);
}

template <core::concepts::Game Game>
typename CompetitiveMctsPlayerGenerator<Game>::BaseMctsPlayer*
CompetitiveMctsPlayerGenerator<Game>::generate_helper(SharedData_sptr& shared_data,
                                                      bool owns_shared_data) {
  return new MctsPlayer(mcts_player_params_, shared_data, owns_shared_data);
}

template <core::concepts::Game Game>
void CompetitiveMctsPlayerGenerator<Game>::parse_args(
    const std::vector<std::string>& args) {
  this->parse_args_helper(make_options_description(), args);
}

// TrainingMctsPlayerGenerator

template <core::concepts::Game Game>
TrainingMctsPlayerGenerator<Game>::TrainingMctsPlayerGenerator()
    : base_t(mcts::kTraining), mcts_player_params_(mcts::kTraining) {}

template <core::concepts::Game Game>
std::string TrainingMctsPlayerGenerator<Game>::get_default_name() const {
  return util::create_string("MCTS-T-%d", mcts_player_params_.num_fast_iters);
}

template <core::concepts::Game Game>
typename TrainingMctsPlayerGenerator<Game>::BaseMctsPlayer*
TrainingMctsPlayerGenerator<Game>::generate_helper(SharedData_sptr& shared_data,
                                                   bool owns_shared_data) {
  return new MctsPlayer(mcts_player_params_, shared_data, owns_shared_data);
}

template <core::concepts::Game Game>
void TrainingMctsPlayerGenerator<Game>::parse_args(
    const std::vector<std::string>& args) {
  this->parse_args_helper(make_options_description(), args);
}

}  // namespace generic
