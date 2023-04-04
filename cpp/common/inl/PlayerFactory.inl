#include <common/PlayerFactory.hpp>

#include <algorithm>

#include <util/BoostUtil.hpp>
#include <util/Exception.hpp>
#include <util/StringUtil.hpp>

namespace common {

template<GameStateConcept GameState>
auto PlayerFactory<GameState>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("PlayerFactory options, for each instance of --player \"...\"");
  return desc
      .template add_option<"type">(po::value<std::string>(&type), "player type. Required")
      .template add_option<"seat">(po::value<int>(&seat), "seat (0 or 1). Random if unspecified")
      ;
}

template<GameStateConcept GameState>
PlayerFactory<GameState>::PlayerFactory(const player_generator_vec_t& generators)
: generators_(generators) {
  // validate that the generator types don't overlap
  std::set<std::string> types;
  for (auto* generator : generators_) {
    for (const auto& type : generator->get_types()) {
      if (types.count(type)) {
        throw util::Exception("PlayerFactory: duplicate type: %s", type.c_str());
      }
      types.insert(type);
    }
  }
}

template<GameStateConcept GameState>
typename PlayerFactory<GameState>::player_generator_seat_vec_t
PlayerFactory<GameState>::parse(const std::vector<std::string>& player_strs) {
  player_generator_seat_vec_t vec;

  for (const auto& player_str : player_strs) {
    vec.push_back(parse_helper(player_str));
  }

  if (vec.empty()) {
    throw util::Exception("At least one --player must be specified");
  }

  if (vec.size() == 1) {
    // if only one player is specified, copy it to fill up the remaining player slots
    const auto& pgs = vec[0];
    while (vec.size() < GameState::kNumPlayers) {
      vec.push_back({pgs.generator, -1});
    }
  }

  return vec;
}

template<GameStateConcept GameState>
void PlayerFactory<GameState>::print_help(const std::vector<std::string>& player_strs)
{
  Params params;
  params.make_options_description().print(std::cout);
  std::cout << "  --... ...             type-specific args, dependent on --type" << std::endl << std::endl;

  std::cout << "For each player, you must pass something like:" << std::endl << std::endl;
  std::cout << "  --player \"--type=MCTS-C <type-specific options...>\"" << std::endl;
  std::cout << "  --player \"--type=TUI --seat=1 <type-specific options...>\"" << std::endl << std::endl;
  std::cout << "If only one --player is specified, it will be copied with --seat=-1 to the remaining player slots.";
  std::cout << std::endl << std::endl;

  std::cout << "The set of legal --type values are:" << std::endl;
  for (auto* generator : generators_) {
    std::cout << "  " << type_str(generator) << ": " << generator->get_description() << std::endl;
  }
  std::cout << std::endl;
  std::cout << "To see the options for a specific --type, pass -h --player \"--type=<type>\"" << std::endl;

  std::vector<bool> used_types(generators_.size(), false);
  for (const std::string &s: player_strs) {
    std::vector<std::string> tokens = util::split(s);
    std::string type = boost_util::get_option_value(tokens, "type");
    for (int g = 0; g < (int)generators_.size(); ++g) {
      if (matches(generators_[g], type)) {
        used_types[g] = true;
        break;
      }
    }
  }

  for (int g = 0; g < (int)generators_.size(); ++g) {
    if (!used_types[g]) continue;

    PlayerGenerator* generator = generators_[g];

    std::ostringstream ss;
    generator->print_help(ss);
    std::string s = ss.str();
    if (!s.empty()) {
      std::cout << std::endl << "--type=" << type_str(generator) << " options:" << std::endl << std::endl;

      std::stringstream ss2(s);
      std::string line;
      while (std::getline(ss2, line, '\n')) {
        std::cout << "  " << line << std::endl;
      }
    }
  }
}

template<GameStateConcept GameState>
std::string PlayerFactory<GameState>::type_str(const PlayerGenerator* generator) {
  std::vector<std::string> types = generator->get_types();
  std::ostringstream ss;
  for (int k = 0; k < (int)types.size(); ++k) {
    if (k > 0) {
      ss << "/";
    }
    ss << types[k];
  }
  return ss.str();
}

template<GameStateConcept GameState>
bool PlayerFactory<GameState>::matches(const PlayerGenerator* generator, const std::string& type) {
  for (const auto& t : generator->get_types()) {
    if (t == type) {
      return true;
    }
  }
  return false;
}

template<GameStateConcept GameState>
typename PlayerFactory<GameState>::player_generator_seat_t
PlayerFactory<GameState>::parse_helper(const std::string& player_str) {
  std::vector<std::string> tokens = util::split(player_str);
  std::string type = boost_util::pop_option_value(tokens, "type");
  std::string seat_str = boost_util::pop_option_value(tokens, "seat");

  int seat = -1;
  if (!seat_str.empty()) {
    seat = std::stoi(seat_str);
    if (seat >= 0 && seat >= GameState::kNumPlayers) {
      throw util::Exception("Invalid seat: %d", seat);
    }
  }

  player_generator_seat_t player_generator_seat;
  player_generator_seat.seat = seat;
  for (auto* generator : generators_) {
    if (matches(generator, type)) {
      generator->parse_args(tokens);
      player_generator_seat.generator = generator;
      break;
    }
  }

  if (!player_generator_seat.generator) {
    throw util::Exception("Unknown player type: %s", type.c_str());
  }
  return player_generator_seat;
}

}  // namespace common
