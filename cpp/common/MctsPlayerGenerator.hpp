#pragma once

#include <string>
#include <vector>

#include <common/AbstractPlayerGenerator.hpp>
#include <common/DataExportingMctsPlayer.hpp>
#include <common/GameStateConcept.hpp>
#include <common/Mcts.hpp>
#include <common/MctsPlayer.hpp>
#include <common/TensorizorConcept.hpp>

namespace common {

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
class MctsPlayerGeneratorBase {
protected:
  using Mcts = common::Mcts<GameState, Tensorizor>;
  struct mcts_play_location_t {
    Mcts* mcts;
    void* play_location;
  };
  using mcts_play_location_vec_t = std::vector<mcts_play_location_t>;

  static mcts_play_location_vec_t mcts_play_locations_;
};

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
class CompetitiveMctsPlayerGenerator :
    public AbstractPlayerGenerator<GameState>,
    public MctsPlayerGeneratorBase<GameState, Tensorizor> {
public:
  using Mcts = common::Mcts<GameState, Tensorizor>;
  using MctsPlayer = common::MctsPlayer<GameState, Tensorizor>;
  using MctsParams = typename Mcts::Params;
  using MctsPlayerParams = typename MctsPlayer::Params;

  CompetitiveMctsPlayerGenerator();
  std::vector<std::string> get_types() const override { return {"MCTS-C", "MCTS-Competitive"}; }
  std::string get_description() const override { return "Competitive MCTS player"; }
  AbstractPlayer<GameState>* generate(void* play_address) override;
  void print_help(std::ostream& s) override { make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args);

protected:
  auto make_options_description() {
    return mcts_params_.make_options_description().add(mcts_player_params_.make_options_description());
  }

  MctsParams mcts_params_;
  MctsPlayerParams mcts_player_params_;
};

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
class TrainingMctsPlayerGenerator :
    public AbstractPlayerGenerator<GameState>,
    public MctsPlayerGeneratorBase<GameState, Tensorizor> {
public:
  using Mcts = common::Mcts<GameState, Tensorizor>;
  using MctsPlayer = common::DataExportingMctsPlayer<GameState, Tensorizor>;
  using MctsParams = typename Mcts::Params;
  using MctsPlayerParams = typename MctsPlayer::Params;
  using TrainingDataWriterParams = typename MctsPlayer::TrainingDataWriterParams;

  TrainingMctsPlayerGenerator();
  std::vector<std::string> get_types() const override { return {"MCTS-T", "MCTS-Training"}; }
  std::string get_description() const override { return "Training MCTS player"; }
  AbstractPlayer<GameState>* generate(void* play_address) override;
  void print_help(std::ostream& s) override { make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;

protected:
  auto make_options_description() {
    return mcts_params_.make_options_description()
      .add(mcts_player_params_.make_options_description())
      .add(writer_params_.make_options_description());
  }

  MctsParams mcts_params_;
  MctsPlayerParams mcts_player_params_;
  TrainingDataWriterParams writer_params_;
};

}  // namespace common

#include <common/inl/MctsPlayerGenerator.inl>
