#pragma once

#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <common/AbstractPlayer.hpp>
#include <common/AbstractPlayerGenerator.hpp>
#include <common/MctsPlayerGenerator.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4PerfectPlayer.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <connect4/OracleGradedMctsPlayer.hpp>
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

  common::AbstractPlayer<c4::GameState>* generate(void* play_address) override;
  void print_help(std::ostream& s) override;
  void parse_args(const std::vector<std::string>& args) override;
  void end_session() override;

protected:
  auto make_options_description() {
    return base_t::make_options_description().add(params_.make_options_description());
  }

  Params params_;
  PerfectOracle* oracle_ = nullptr;
  OracleGrader* grader_ = nullptr;
};

using TrainingMctsPlayerGenerator = common::TrainingMctsPlayerGenerator<c4::GameState, c4::Tensorizor>;

}  // namespace c4

#include <connect4/inl/C4MctsPlayerGenerator.inl>