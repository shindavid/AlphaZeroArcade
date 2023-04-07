#include <connect4/C4MctsPlayerGenerator.hpp>

namespace c4 {

inline CompetitiveMctsPlayerGenerator::~CompetitiveMctsPlayerGenerator() {
  if (grader_) delete grader_;
  if (oracle_) delete oracle_;
}

inline common::AbstractPlayer<c4::GameState>* CompetitiveMctsPlayerGenerator::generate(void* play_address) {
  if (params_.grade_moves) {
    if (oracle_ == nullptr) {
      oracle_ = new PerfectOracle();
      grader_ = new OracleGrader(oracle_);
    }

    for (const auto& mpl : mcts_play_locations_) {
      if (mpl.play_location == play_address && mpl.mcts->params() == mcts_params_) {
        return new OracleGradedMctsPlayer(grader_, mcts_player_params_, mpl.mcts);
      }
    }

    auto player = new OracleGradedMctsPlayer(grader_, mcts_player_params_, mcts_params_);
    Mcts* mcts = player->get_mcts();
    mcts_play_locations_.push_back({mcts, play_address});
    return player;
  } else {
    return generate(play_address);
  }
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
