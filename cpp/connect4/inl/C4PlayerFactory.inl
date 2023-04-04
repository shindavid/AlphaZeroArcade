#include <connect4/C4PlayerFactory.hpp>

#include <connect4/C4Tensorizor.hpp>
#include <util/BoostUtil.hpp>
#include <util/Exception.hpp>
#include <util/StringUtil.hpp>

namespace common {

template<GameStateConcept GameState, PlayerWidgetConcept<GameState>... Widgets>
PlayerFactory<GameState, Widgets...>* PlayerFactory<GameState, Widgets...>::PlayerFactory::instance_ = nullptr;

template<GameStateConcept GameState>
auto PlayerFactoryBase<GameState>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("PlayerFactory options, for each instance of --player \"...\"");
  return desc
      .template add_option<"type">(po::value<std::string>(&type), "player type. Required")
      .template add_option<"seat">(po::value<int>(&seat), "seat (0 or 1). Random if unspecified")
      ;
}

template<GameStateConcept GameState, PlayerWidgetConcept<GameState>... Widgets>
typename PlayerFactoryBase<GameState>::player_generator_seat_vec_t
PlayerFactory<GameState, Widgets...>::parse(const std::vector<std::string>& player_strs) {
  player_generator_seat_vec_t vec;

  PlayerFactory* factory = instance();
  for (const auto& player_str : player_strs) {
    vec.push_back(factory->parse_helper(player_str));
  }

  return vec;
}

template<GameStateConcept GameState, PlayerWidgetConcept<GameState>... Widgets>
void PlayerFactory<GameState, Widgets...>::print_help(const std::vector<std::string>& player_strs)
{
  typename base_t::Params params;
  params.make_options_description().print(std::cout);

  std::cout << std::endl;
  std::cout << "For each player, you must pass something like:" << std::endl;
  std::cout << "  --player \"--type=MCTS-C <type-specific options...>\"" << std::endl;
  std::cout << "  --player \"--type=TUI --seat=1 <type-specific options...>\"" << std::endl << std::endl;

  std::cout << "The set of legal --type values are:" << std::endl;
  (print_description<Widgets>(), ...);

  bool used_types[kNumWidgets] = {};

  for (const std::string &s: player_strs) {
    std::vector<std::string> tokens = util::split(s);
    std::string type = boost_util::get_option_value(tokens, "type");
    int w = 0;
    (matches_type<Widgets>(type, used_types[w++]), ...);
  }

  PlayerFactory* factory = instance();
  int w = 0;
  (factory->print_type_options<Widgets>(used_types[w++]), ...);
}

template<GameStateConcept GameState, PlayerWidgetConcept<GameState>... Widgets>
PlayerFactory<GameState, Widgets...>* PlayerFactory<GameState, Widgets...>::instance() {
  if (!instance_) {
    instance_ = new PlayerFactory();
  }
  return instance_;
}

template<GameStateConcept GameState, PlayerWidgetConcept<GameState>... Widgets>
typename PlayerFactoryBase<GameState>::player_generator_seat_t
PlayerFactory<GameState, Widgets...>::parse_helper(const std::string& player_str) {
  std::vector<std::string> tokens = util::split(player_str);
  std::string type = boost_util::pop_option_value(tokens, "type");
  std::string seat_str = boost_util::pop_option_value(tokens, "seat");

  int seat = -1;
  if (!seat_str.empty()) {
    seat = std::stoi(seat_str);
    if (seat >= 0 && seat >= kNumPlayers) {
      throw util::Exception("Invalid seat: %d", seat);
    }
  }

  player_generator_seat_t player_generator_seat;
  player_generator_seat.seat = seat;
  bool matched = false;
  (set_generator<Widgets>(type, tokens, player_generator_seat, matched), ...);

  if (!matched) {
    throw util::Exception("Unknown player type: %s", type.c_str());
  }
  return player_generator_seat;
}

}  // namespace common

namespace c4 {


}  // namespace c4
