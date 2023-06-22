#include <common/players/MctsPlayerGenerator.hpp>

namespace common {

// MctsPlayerGeneratorBase

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename MctsPlayerGeneratorBase<GameState, Tensorizor>::mcts_map_t
    MctsPlayerGeneratorBase<GameState, Tensorizor>::mcts_cache_;

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
core::AbstractPlayer<GameState>* MctsPlayerGeneratorBase<GameState, Tensorizor>::generate(
    core::game_thread_id_t game_thread_id)
{
  mcts_vec_t& vec = mcts_cache_[game_thread_id];
  for (Mcts* mcts : vec) {
    if (mcts->params() == mcts_params_) {
      return generate_from_mcts(mcts);
    }
  }

  auto player = generate_from_scratch();
  Mcts* mcts = player->get_mcts();
  vec.push_back(mcts);
  return player;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void MctsPlayerGeneratorBase<GameState, Tensorizor>::end_session() {
  Mcts::end_session();
}

// CompetitiveMctsPlayerGenerator

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::CompetitiveMctsPlayerGenerator()
: base_t(Mcts::kCompetitive)
, mcts_player_params_(MctsPlayer::kCompetitive) {}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::BaseMctsPlayer*
CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::generate_from_scratch()
{
  return new MctsPlayer(mcts_player_params_, this->mcts_params_);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::BaseMctsPlayer*
CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::generate_from_mcts(Mcts* mcts)
{
  return new MctsPlayer(mcts_player_params_, mcts);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::parse_args(const std::vector<std::string>& args) {
  this->parse_args_helper(make_options_description(), args);
}

// TrainingMctsPlayerGenerator

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
TrainingMctsPlayerGenerator<GameState, Tensorizor>::TrainingMctsPlayerGenerator()
: base_t(Mcts::kTraining)
, mcts_player_params_(MctsPlayer::kTraining) {}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename TrainingMctsPlayerGenerator<GameState, Tensorizor>::BaseMctsPlayer*
TrainingMctsPlayerGenerator<GameState, Tensorizor>::generate_from_scratch()
{
  return new MctsPlayer(writer_params_, mcts_player_params_, this->mcts_params_);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename TrainingMctsPlayerGenerator<GameState, Tensorizor>::BaseMctsPlayer*
TrainingMctsPlayerGenerator<GameState, Tensorizor>::generate_from_mcts(Mcts* mcts)
{
  return new MctsPlayer(writer_params_, mcts_player_params_, mcts);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void TrainingMctsPlayerGenerator<GameState, Tensorizor>::parse_args(const std::vector<std::string>& args) {
  this->parse_args_helper(make_options_description(), args);
}

}  // namespace common
