#include <connect4/C4MctsPlayerGenerator.hpp>

namespace c4 {

inline CompetitiveMctsPlayerGenerator::~CompetitiveMctsPlayerGenerator() {
  if (grader_) delete grader_;
  if (oracle_) delete oracle_;
}

inline CompetitiveMctsPlayerGenerator::BaseMctsPlayer* CompetitiveMctsPlayerGenerator::generate_from_scratch() {
  return new OracleGradedMctsPlayer(grader_, mcts_player_params_, mcts_params_);
}

inline CompetitiveMctsPlayerGenerator::BaseMctsPlayer* CompetitiveMctsPlayerGenerator::generate_from_mcts(Mcts* mcts) {
  return new OracleGradedMctsPlayer(grader_, mcts_player_params_, mcts);
}

inline void CompetitiveMctsPlayerGenerator::print_help(std::ostream& s) {
  make_options_description().print(s);
}

inline void CompetitiveMctsPlayerGenerator::parse_args(const std::vector<std::string>& args) {
  this->parse_args_helper(make_options_description(), args);
}

inline void CompetitiveMctsPlayerGenerator::end_session() {
  if (grader_) grader_->dump();
}

}  // namespace c4
