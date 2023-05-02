#include <common/players/MctsPlayerGenerator.hpp>

namespace common {

// MctsPlayerGeneratorBase

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename MctsPlayerGeneratorBase<GameState, Tensorizor>::mcts_map_t
    MctsPlayerGeneratorBase<GameState, Tensorizor>::mcts_cache_;

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
AbstractPlayer<GameState>* MctsPlayerGeneratorBase<GameState, Tensorizor>::generate(game_thread_id_t game_thread_id) {
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

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void MctsPlayerGeneratorBase<GameState, Tensorizor>::validate_params() {
  if (!(mcts_params_.nnet_filename.empty() ^ mcts_params_.uniform_model)) {
    throw util::CleanException(
        "Invalid MCTS player: exactly one of --nnet-filename or --uniform-model must be specified");
  }
}

// CompetitiveMctsPlayerGenerator

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::CompetitiveMctsPlayerGenerator()
: base_t(Mcts::kCompetitive)
, mcts_player_params_(MctsPlayer::kCompetitive) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::BaseMctsPlayer*
CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::generate_from_scratch()
{
  return new MctsPlayer(mcts_player_params_, this->mcts_params_);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::BaseMctsPlayer*
CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::generate_from_mcts(Mcts* mcts)
{
  return new MctsPlayer(mcts_player_params_, mcts);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::parse_args(const std::vector<std::string>& args) {
  this->parse_args_helper(make_options_description(), args);
  this->validate_params();
}

// TrainingMctsPlayerGenerator

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
TrainingMctsPlayerGenerator<GameState, Tensorizor>::TrainingMctsPlayerGenerator()
: base_t(Mcts::kTraining)
, mcts_player_params_(MctsPlayer::kTraining) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename TrainingMctsPlayerGenerator<GameState, Tensorizor>::BaseMctsPlayer*
TrainingMctsPlayerGenerator<GameState, Tensorizor>::generate_from_scratch()
{
  return new MctsPlayer(writer_params_, mcts_player_params_, this->mcts_params_);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename TrainingMctsPlayerGenerator<GameState, Tensorizor>::BaseMctsPlayer*
TrainingMctsPlayerGenerator<GameState, Tensorizor>::generate_from_mcts(Mcts* mcts)
{
  return new MctsPlayer(writer_params_, mcts_player_params_, mcts);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void TrainingMctsPlayerGenerator<GameState, Tensorizor>::parse_args(const std::vector<std::string>& args) {
  this->parse_args_helper(make_options_description(), args);
  this->validate_params();
}

}  // namespace common
