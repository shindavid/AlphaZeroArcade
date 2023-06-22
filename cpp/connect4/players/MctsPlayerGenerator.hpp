#pragma once

#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <core/AbstractPlayer.hpp>
#include <core/AbstractPlayerGenerator.hpp>
#include <common/players/MctsPlayerGenerator.hpp>
#include <connect4/GameState.hpp>
#include <connect4/players/PerfectPlayer.hpp>
#include <connect4/Tensorizor.hpp>
#include <connect4/players/OracleGradedMctsPlayer.hpp>
#include <util/BoostUtil.hpp>

namespace c4 {

class CompetitiveMctsPlayerGenerator : public common::CompetitiveMctsPlayerGenerator<c4::GameState, c4::Tensorizor> {
public:
  using base_t = common::CompetitiveMctsPlayerGenerator<c4::GameState, c4::Tensorizor>;

  struct Params {
    bool grade_moves;

    auto make_options_description() {
      namespace po = boost::program_options;
      namespace po2 = boost_util::program_options;

      po2::options_description desc("c4::CompetitiveMctsPlayerGenerator options");
      return desc
          .template add_option<"grade-moves">(po::bool_switch(&grade_moves),
                                              "use perfect oracle to report % of moves that were correct")
          ;
    }
  };

  ~CompetitiveMctsPlayerGenerator() override;

  void print_help(std::ostream& s) override;
  void parse_args(const std::vector<std::string>& args) override;
  void end_session() override;

protected:
  auto make_options_description() {
    return base_t::make_options_description().add(params_.make_options_description());
  }

  BaseMctsPlayer* generate_from_scratch() override;
  BaseMctsPlayer* generate_from_mcts(Mcts* mcts) override;

  Params params_;
  PerfectOracle* oracle_ = nullptr;
  OracleGrader* grader_ = nullptr;
};

using TrainingMctsPlayerGenerator = common::TrainingMctsPlayerGenerator<c4::GameState, c4::Tensorizor>;

}  // namespace c4

#include <connect4/players/inl/MctsPlayerGenerator.inl>
