#pragma once

#include <map>
#include <string>
#include <vector>

#include <common/AbstractPlayerGenerator.hpp>
#include <common/GameStateConcept.hpp>
#include <common/Mcts.hpp>
#include <common/TensorizorConcept.hpp>
#include <common/players/DataExportingMctsPlayer.hpp>
#include <common/players/MctsPlayer.hpp>

namespace common {

class TrainingMctsPlayerGeneratorBase {};

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
class MctsPlayerGeneratorBase : public AbstractPlayerGenerator<GameState> {
public:
  using Mcts = common::Mcts<GameState, Tensorizor>;
  using MctsParams = typename Mcts::Params;
  using BaseMctsPlayer = common::MctsPlayer<GameState, Tensorizor>;

  MctsPlayerGeneratorBase(Mcts::DefaultParamsType type) : mcts_params_(type) {}

  /*
   * If this generator already generated a player for the given game_thread_id, dispatches to generate_from_mcts(),
   * passing in the Mcts* of that previous player. Otherwise, dispatches to generate_from_scratch().
   */
  AbstractPlayer<GameState>* generate(game_thread_id_t game_thread_id) override;

protected:
  virtual BaseMctsPlayer* generate_from_scratch() = 0;
  virtual BaseMctsPlayer* generate_from_mcts(Mcts* mcts) = 0;

  void validate_params();

  using mcts_vec_t = std::vector<Mcts*>;
  using mcts_map_t = std::map<game_thread_id_t, mcts_vec_t>;

  static mcts_map_t mcts_cache_;

  MctsParams mcts_params_;
};

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
class CompetitiveMctsPlayerGenerator :
    public MctsPlayerGeneratorBase<GameState, Tensorizor> {
public:
  using base_t = MctsPlayerGeneratorBase<GameState, Tensorizor>;
  using BaseMctsPlayer = typename base_t::BaseMctsPlayer;
  using Mcts = common::Mcts<GameState, Tensorizor>;
  using MctsPlayer = common::MctsPlayer<GameState, Tensorizor>;
  using MctsPlayerParams = typename MctsPlayer::Params;

  CompetitiveMctsPlayerGenerator();
  std::vector<std::string> get_types() const override { return {"MCTS-C", "MCTS-Competitive"}; }
  std::string get_description() const override { return "Competitive MCTS player"; }
  void print_help(std::ostream& s) override { make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;

protected:
  auto make_options_description() {
    return this->mcts_params_.make_options_description().add(mcts_player_params_.make_options_description());
  }

  BaseMctsPlayer* generate_from_scratch() override;
  BaseMctsPlayer* generate_from_mcts(Mcts* mcts) override;

  MctsPlayerParams mcts_player_params_;
};

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
class TrainingMctsPlayerGenerator :
    public MctsPlayerGeneratorBase<GameState, Tensorizor>, public TrainingMctsPlayerGeneratorBase {
public:
  using base_t = MctsPlayerGeneratorBase<GameState, Tensorizor>;
  using BaseMctsPlayer = typename base_t::BaseMctsPlayer;
  using Mcts = common::Mcts<GameState, Tensorizor>;
  using MctsPlayer = common::DataExportingMctsPlayer<GameState, Tensorizor>;
  using MctsPlayerParams = typename MctsPlayer::Params;
  using TrainingDataWriterParams = typename MctsPlayer::TrainingDataWriterParams;

  TrainingMctsPlayerGenerator();
  std::vector<std::string> get_types() const override { return {"MCTS-T", "MCTS-Training"}; }
  std::string get_description() const override { return "Training MCTS player"; }
  void print_help(std::ostream& s) override { make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;

protected:
  auto make_options_description() {
    return this->mcts_params_.make_options_description()
      .add(mcts_player_params_.make_options_description())
      .add(writer_params_.make_options_description());
  }

  BaseMctsPlayer* generate_from_scratch() override;
  BaseMctsPlayer* generate_from_mcts(Mcts* mcts) override;

  MctsPlayerParams mcts_player_params_;
  TrainingDataWriterParams writer_params_;
};

}  // namespace common

#include <common/players/inl/MctsPlayerGenerator.inl>

