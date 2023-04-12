#pragma once

#include <map>
#include <string>
#include <utility>

#include <boost/program_options.hpp>

#include <common/AbstractPlayer.hpp>
#include <common/AbstractPlayerGenerator.hpp>
#include <common/GameStateConcept.hpp>
#include <common/HumanTuiPlayer.hpp>
#include <common/Mcts.hpp>
#include <common/MctsPlayer.hpp>
#include <common/RandomPlayer.hpp>
#include <util/CppUtil.hpp>
#include <util/MetaProgramming.hpp>

/*
 * The PlayerFactory is a template class that facilitates the creation of player objects from command line arguments.
 *
 * See cpp/connect4/C4PlayerFactor.hpp for an example of how to use this class.
 */
namespace common {

template<GameStateConcept GameState_>
class PlayerFactory {
public:
  using GameState = GameState_;
  using Player = AbstractPlayer<GameState>;
  using PlayerGenerator = AbstractPlayerGenerator<GameState>;
  using PlayerGeneratorCreatorBase = common::PlayerGeneratorCreatorBase<GameState>;

  struct player_generator_seat_t {
    PlayerGenerator* generator = nullptr;
    int seat = -1;
  };
  using player_generator_seat_vec_t = std::vector<player_generator_seat_t>;
  using player_generator_vec_t = std::vector<PlayerGenerator*>;
  using player_generator_creator_vec_t = std::vector<PlayerGeneratorCreatorBase*>;

  struct Params {
    auto make_options_description();

    std::string type;
    std::string name;
    std::string copy_from;
    int seat = -1;
  };

  /*
   * The constructor takes a vector of PlayerGenerator objects. Each PlayerGenerator provides a recipe for how to
   * determine whether a set of cmdline tokens match that generator, and if so, how to create a player object.
   *
   * From this, the PlayerFactory generates the appropriate help message, with player-specific details printed only if
   * the associated type is passed via --player "--type=..."
   */
  PlayerFactory(const player_generator_creator_vec_t& creators);

  player_generator_seat_vec_t parse(const std::vector<std::string>& player_strs);
  void print_help(const std::vector<std::string>& player_strs);

private:
  static std::string type_str(const PlayerGenerator* generator);
  static bool matches(const PlayerGenerator* generator, const std::string& type);
  PlayerGenerator* parse_helper(const std::string& player_str, const std::string& name,
                                const std::vector<std::string>& tokens);

  player_generator_creator_vec_t creators_;
  std::map<std::string, std::vector<std::string>> name_map_;
};

}  // namespace common

#include <common/inl/PlayerFactory.inl>
