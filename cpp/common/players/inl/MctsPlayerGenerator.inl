#include <common/players/MctsPlayerGenerator.hpp>

namespace common {

// MctsPlayerGeneratorBase

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename MctsPlayerGeneratorBase<GameState, Tensorizor>::mcts_map_t
    MctsPlayerGeneratorBase<GameState, Tensorizor>::mcts_cache_;

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
AbstractPlayer<GameState>* MctsPlayerGeneratorBase<GameState, Tensorizor>::generate(game_thread_id_t game_thread_id) {
  mcts_map_value_t& value = mcts_cache_[game_thread_id];
  for (Mcts* mcts : value.vec) {
    if (mcts->params() == mcts_params_) {
      auto player = generate_from_mcts(mcts);
      util::clean_assert(value.shared_data, "unexpected null shared_data");
      player->set_shared_data(value.shared_data);
      return player;
    }
  }

  auto player = generate_from_scratch();
  if (value.shared_data == nullptr) {
    value.shared_data = player->init_shared_data();
  } else {
    player->set_shared_data(value.shared_data);
  }
  Mcts* mcts = player->get_mcts();
  value.vec.push_back(mcts);
  return player;
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
}

}  // namespace common
