#include <common/players/MctsPlayerGenerator.hpp>

#include <mcts/Constants.hpp>

namespace common {

// MctsPlayerGeneratorBase

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename MctsPlayerGeneratorBase<GameState, Tensorizor>::manager_map_t
    MctsPlayerGeneratorBase<GameState, Tensorizor>::manager_cache_;

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
core::AbstractPlayer<GameState>* MctsPlayerGeneratorBase<GameState, Tensorizor>::generate(
    core::game_thread_id_t game_thread_id)
{
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

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void MctsPlayerGeneratorBase<GameState, Tensorizor>::end_session() {
  MctsManager::end_session();
}

// CompetitiveMctsPlayerGenerator

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::CompetitiveMctsPlayerGenerator()
: base_t(mcts::kCompetitive)
, mcts_player_params_(mcts::kCompetitive) {}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::BaseMctsPlayer*
CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::generate_from_scratch()
{
  return new MctsPlayer(mcts_player_params_, this->manager_params_);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::BaseMctsPlayer*
CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::generate_from_manager(MctsManager* manager)
{
  return new MctsPlayer(mcts_player_params_, manager);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::parse_args(const std::vector<std::string>& args) {
  this->parse_args_helper(make_options_description(), args);
}

// TrainingMctsPlayerGenerator

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
TrainingMctsPlayerGenerator<GameState, Tensorizor>::TrainingMctsPlayerGenerator()
: base_t(mcts::kTraining)
, mcts_player_params_(mcts::kTraining) {}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename TrainingMctsPlayerGenerator<GameState, Tensorizor>::BaseMctsPlayer*
TrainingMctsPlayerGenerator<GameState, Tensorizor>::generate_from_scratch()
{
  return new MctsPlayer(writer_params_, mcts_player_params_, this->manager_params_);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename TrainingMctsPlayerGenerator<GameState, Tensorizor>::BaseMctsPlayer*
TrainingMctsPlayerGenerator<GameState, Tensorizor>::generate_from_manager(MctsManager* manager)
{
  return new MctsPlayer(writer_params_, mcts_player_params_, manager);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void TrainingMctsPlayerGenerator<GameState, Tensorizor>::parse_args(const std::vector<std::string>& args) {
  this->parse_args_helper(make_options_description(), args);
}

}  // namespace common
