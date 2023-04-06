#include <connect4/C4HumanTuiPlayerGenerator.hpp>

#include <boost/program_options.hpp>

namespace c4 {

inline common::AbstractPlayer<c4::GameState>* HumanTuiPlayerGenerator::generate(void* play_address) {
  if (params_.cheat_mode) {
    return new CheatingHumanTuiPlayer();
  } else {
    return new common::HumanTuiPlayer<GameState>();
  }
}

inline void HumanTuiPlayerGenerator::parse_args(const std::vector<std::string>& args) {
  namespace po = boost::program_options;
  po::variables_map vm;
  po::store(po::command_line_parser(args).options(params_.make_options_description()).run(), vm);
  po::notify(vm);
}

}  // namespace c4
