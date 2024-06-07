#include <generic_players/MctsPlayerGenerator.hpp>

#include <mcts/Constants.hpp>

namespace generic {

// MctsPlayerGeneratorBase

template <core::concepts::Game Game>
typename MctsPlayerGeneratorBase<Game>::manager_map_t
    MctsPlayerGeneratorBase<Game>::manager_cache_;

template <core::concepts::Game Game>
core::AbstractPlayer<GameState>* MctsPlayerGeneratorBase<Game>::generate(
    core::game_thread_id_t game_thread_id) {
  manager_vec_t& vec = manager_cache_[game_thread_id];
  for (MctsManager* manager : vec) {
    if (manager->params() == manager_params_) {
      return generate_from_manager(manager);
    }
  }

  auto player = generate_from_scratch();
  MctsManager* manager = player->get_mcts_manager();
  vec.push_back(manager);
  return player;
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
CompetitiveMctsPlayerGenerator<Game>::generate_from_scratch() {
  return new MctsPlayer(mcts_player_params_, this->manager_params_);
}

template <core::concepts::Game Game>
typename CompetitiveMctsPlayerGenerator<Game>::BaseMctsPlayer*
CompetitiveMctsPlayerGenerator<Game>::generate_from_manager(MctsManager* manager) {
  return new MctsPlayer(mcts_player_params_, manager);
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
TrainingMctsPlayerGenerator<Game>::generate_from_scratch() {
  return new MctsPlayer(writer_params_, mcts_player_params_, this->manager_params_);
}

template <core::concepts::Game Game>
typename TrainingMctsPlayerGenerator<Game>::BaseMctsPlayer*
TrainingMctsPlayerGenerator<Game>::generate_from_manager(MctsManager* manager) {
  return new MctsPlayer(writer_params_, mcts_player_params_, manager);
}

template <core::concepts::Game Game>
void TrainingMctsPlayerGenerator<Game>::parse_args(
    const std::vector<std::string>& args) {
  this->parse_args_helper(make_options_description(), args);
}

}  // namespace generic
