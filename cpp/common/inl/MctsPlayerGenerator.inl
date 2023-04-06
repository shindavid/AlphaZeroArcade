#include <common/MctsPlayerGenerator.hpp>

namespace common {

// MctsPlayerGeneratorBase

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename MctsPlayerGeneratorBase<GameState, Tensorizor>::mcts_play_location_vec_t
    MctsPlayerGeneratorBase<GameState, Tensorizor>::mcts_play_locations_;

// CompetitiveMctsPlayerGenerator

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::CompetitiveMctsPlayerGenerator()
: mcts_params_(Mcts::kCompetitive)
, mcts_player_params_(MctsPlayer::kCompetitive) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
AbstractPlayer<GameState>* CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::generate(void* play_address) {
  using base_t = MctsPlayerGeneratorBase<GameState, Tensorizor>;
  for (const auto& mpl : base_t::mcts_play_locations_) {
    if (mpl.play_location == play_address && mpl.mcts->params() == mcts_params_) {
      return new MctsPlayer(mcts_player_params_, mpl.mcts);
    }
  }

  auto player = new MctsPlayer(mcts_player_params_, mcts_params_);
  Mcts* mcts = player->get_mcts();
  base_t::mcts_play_locations_.push_back({mcts, play_address});
  return player;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void CompetitiveMctsPlayerGenerator<GameState, Tensorizor>::parse_args(const std::vector<std::string>& args) {
  namespace po = boost::program_options;
  po::variables_map vm;
  po::store(po::command_line_parser(args).options(make_options_description()).run(), vm);
  po::notify(vm);
}

// TrainingMctsPlayerGenerator

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
TrainingMctsPlayerGenerator<GameState, Tensorizor>::TrainingMctsPlayerGenerator()
: mcts_params_(Mcts::kTraining)
, mcts_player_params_(MctsPlayer::kTraining) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
AbstractPlayer<GameState>* TrainingMctsPlayerGenerator<GameState, Tensorizor>::generate(void* play_address) {
  using base_t = MctsPlayerGeneratorBase<GameState, Tensorizor>;
  for (const auto& mpl : base_t::mcts_play_locations_) {
    if (mpl.play_location == play_address && mpl.mcts->params() == mcts_params_) {
      return new MctsPlayer(writer_params_, mcts_player_params_, mpl.mcts);
    }
  }

  auto player = new MctsPlayer(writer_params_, mcts_player_params_, mcts_params_);
  Mcts* mcts = player->get_mcts();
  base_t::mcts_play_locations_.push_back({mcts, play_address});
  return player;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void TrainingMctsPlayerGenerator<GameState, Tensorizor>::parse_args(const std::vector<std::string>& args) {
  namespace po = boost::program_options;
  po::variables_map vm;
  po::store(po::command_line_parser(args).options(make_options_description()).run(), vm);
  po::notify(vm);
}

}  // namespace common
