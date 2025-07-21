#include <core/PlayerFactory.hpp>

#include <util/Asserts.hpp>
#include <util/BoostUtil.hpp>
#include <util/Exception.hpp>
#include <util/StringUtil.hpp>

namespace core {

template <concepts::Game Game>
auto PlayerFactory<Game>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("PlayerFactory options, for each instance of --player \"...\"");
  return desc.template add_option<"type">(po::value<std::string>(&type), "required")
      .template add_option<"name">(po::value<std::string>(&name),
                                   "if unspecified, then a default name is chosen")
      .template add_option<"copy-from">(
          po::value<std::string>(&copy_from),
          "copy everything but --name and --seat from the --player with this name")
      .template add_option<"seat">(po::value<int>(&seat),
                                   "zero-indexed seat (0=first-player, 1=second-player, etc.). If "
                                   "unspecified, then the seat is selected "
                                   "randomly among the available seats for the first game. For "
                                   "subsequent games, we round-robin through the "
                                   "set of N! possible seat assignments, where N is the number of "
                                   "players with unspecified seats.");
}

template <concepts::Game Game>
PlayerFactory<Game>::PlayerFactory(const player_subfactory_vec_t& subfactories)
    : subfactories_(subfactories) {
  // validate that the generator types don't overlap
  std::set<std::string> types;
  for (auto* subfactory : subfactories_) {
    auto* generator = subfactory->create(nullptr);
    for (const auto& type : generator->get_types()) {
      if (types.count(type)) {
        throw util::Exception("PlayerFactory: duplicate type: {}", type);
      }
      types.insert(type);
    }
    delete generator;
  }
}

template <concepts::Game Game>
typename PlayerFactory<Game>::player_generator_seat_vec_t PlayerFactory<Game>::parse(
    const std::vector<std::string>& player_strs) {
  RELEASE_ASSERT(server_ != nullptr,
                       "PlayerFactory::parse() called without a server");
  player_generator_seat_vec_t vec;

  for (const auto& player_str : player_strs) {
    std::vector<std::string> tokens = util::split(player_str);

    std::string name = boost_util::pop_option_value(tokens, "name");
    std::string seat_str = boost_util::pop_option_value(tokens, "seat");

    int seat = -1;
    if (!seat_str.empty()) {
      seat = std::stoi(seat_str);
      CLEAN_ASSERT(seat < Game::Constants::kNumPlayers,
                         "Invalid seat ({}) in --player \"{}\"", seat, player_str);
    }

    PlayerGeneratorSeat player_generator_seat;
    player_generator_seat.generator = parse_helper(player_str, name, tokens);
    player_generator_seat.seat = seat;

    vec.push_back(player_generator_seat);
  }

  return vec;
}

template <concepts::Game Game>
void PlayerFactory<Game>::print_help(const std::vector<std::string>& player_strs) {
  Params params;
  std::cout << params.make_options_description();
  std::cout << "  --... ...             type-specific args, dependent on --type" << std::endl
            << std::endl;

  std::cout << "For each player, you must pass something like:" << std::endl << std::endl;
  std::cout << "  --player \"--type=MCTS-C --name=CPU <type-specific options...>\"" << std::endl;
  std::cout << "  --player \"--type=TUI --name=Human --seat=1 <type-specific options...>\""
            << std::endl
            << std::endl;

  std::cout << "The set of legal --type values are:" << std::endl;

  player_generator_vec_t generators;
  for (auto* subfactory : subfactories_) {
    generators.push_back(subfactory->create(nullptr));
  }
  for (auto* generator : generators) {
    std::cout << "  " << type_str(generator) << ": " << generator->get_description() << std::endl;
  }
  std::cout << std::endl;
  std::cout << "To see the options for a specific --type, pass -h --player \"--type=<type>\""
            << std::endl;

  std::vector<bool> used_types(generators.size(), false);
  for (const std::string& s : player_strs) {
    std::vector<std::string> tokens = util::split(s);
    std::string type = boost_util::get_option_value(tokens, "type");
    for (int g = 0; g < (int)generators.size(); ++g) {
      if (matches(generators[g], type)) {
        used_types[g] = true;
        break;
      }
    }
  }

  for (int g = 0; g < (int)generators.size(); ++g) {
    if (!used_types[g]) continue;

    PlayerGenerator* generator = generators[g];

    std::ostringstream ss;
    generator->print_help(ss);
    std::string s = ss.str();
    if (!s.empty()) {
      std::cout << std::endl
                << "--type=" << type_str(generator) << " options:" << std::endl
                << std::endl;

      std::stringstream ss2(s);
      std::string line;
      while (std::getline(ss2, line, '\n')) {
        std::cout << "  " << line << std::endl;
      }
    }
  }

  for (auto* generator : generators) {
    delete generator;
  }
}

template <concepts::Game Game>
std::string PlayerFactory<Game>::type_str(const PlayerGenerator* generator) {
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

template <concepts::Game Game>
bool PlayerFactory<Game>::matches(const PlayerGenerator* generator, const std::string& type) {
  for (const auto& t : generator->get_types()) {
    if (t == type) {
      return true;
    }
  }
  return false;
}

template <concepts::Game Game>
typename PlayerFactory<Game>::PlayerGenerator* PlayerFactory<Game>::parse_helper(
    const std::string& player_str, const std::string& name,
    const std::vector<std::string>& orig_tokens) {
  std::vector<std::string> tokens = orig_tokens;

  std::string type = boost_util::pop_option_value(tokens, "type");
  std::string copy_from = boost_util::pop_option_value(tokens, "copy-from");

  if (!copy_from.empty()) {
    if (!type.empty()) {
      throw util::Exception("Invalid usage of --copy-from with --type in --player \"{}\"",
                            player_str);
    }
    CLEAN_ASSERT(name_map_.count(copy_from), "Invalid --copy-from in --player \"{}\"",
                       player_str);
    return parse_helper(player_str, name, name_map_.at(copy_from));
  }

  CLEAN_ASSERT(!type.empty(), "Must specify --type or --copy-from in --player \"{}\"",
                     player_str);
  if (!name.empty()) {
    CLEAN_ASSERT(!name_map_.count(name), "Duplicate --name \"{}\"", name);
    name_map_[name] = orig_tokens;
  }
  PlayerGenerator* matched_generator = nullptr;
  for (auto* subfactory : subfactories_) {
    auto* generator = subfactory->create(server_);
    if (matches(generator, type)) {
      CLEAN_ASSERT(matched_generator == nullptr, "Type {}: multiple matches", type);
      matched_generator = generator;
      continue;
    }
    delete generator;
  }

  CLEAN_ASSERT(matched_generator != nullptr, "Unknown type in --player \"{}\"", player_str);

  matched_generator->set_name(name);
  matched_generator->parse_args(tokens);
  return matched_generator;
}

}  // namespace core
