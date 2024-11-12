#include <generic_players/MctsPlayerGenerator.hpp>

#include <mcts/Constants.hpp>

namespace generic {

// MctsPlayerGeneratorBase

template <core::concepts::Game Game>
typename MctsPlayerGeneratorBase<Game>::manager_map_t
    MctsPlayerGeneratorBase<Game>::manager_cache_;

template <core::concepts::Game Game>
core::AbstractPlayer<Game>* MctsPlayerGeneratorBase<Game>::generate(
    core::game_thread_id_t game_thread_id) {
  manager_vec_t& vec = manager_cache_[game_thread_id];
  for (MctsManager* manager : vec) {
    if (manager->params() == manager_params_) {
      return generate_helper(manager, false);
    }
  }

  // TODO: consider using std::shared_ptr here to make object-ownership logic more robust.
  // If we do so, we should rethink the "player-data" mechanism in MctsManager, so that the
  // MctsPlayer::SharedData object is also able to use std::shared_ptr.
  MctsManager* manager = new MctsManager(this->manager_params_);
  vec.push_back(manager);
  return generate_helper(manager, true);
}

template <core::concepts::Game Game>
void MctsPlayerGeneratorBase<Game>::end_session() {
  for (auto& pair : manager_cache_) {
    for (MctsManager* manager : pair.second) {
      manager->end_session();
    }
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
CompetitiveMctsPlayerGenerator<Game>::generate_helper(MctsManager* manager, bool owns_manager) {
  return new MctsPlayer(mcts_player_params_, manager, owns_manager);
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
TrainingMctsPlayerGenerator<Game>::generate_helper(MctsManager* manager, bool owns_manager) {
  return new MctsPlayer(writer_params_, mcts_player_params_, manager, owns_manager);
}

template <core::concepts::Game Game>
void TrainingMctsPlayerGenerator<Game>::parse_args(
    const std::vector<std::string>& args) {
  this->parse_args_helper(make_options_description(), args);
}

}  // namespace generic
