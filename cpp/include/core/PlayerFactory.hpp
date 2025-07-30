#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/AbstractPlayerGenerator.hpp"
#include "core/GameServerBase.hpp"
#include "core/concepts/Game.hpp"

#include <boost/program_options.hpp>

#include <map>
#include <string>

namespace core {

template <concepts::Game Game>
class PlayerSubfactoryBase {
 public:
  virtual ~PlayerSubfactoryBase() = default;
  virtual AbstractPlayerGenerator<Game>* create(GameServerBase*) = 0;
};

/*
 * PlayerSubfactory is a helper class used by PlayerFactory. Each PlayerSubfactory is associated
 * with a particular player type.
 *
 * A PlayerFactory in turn is associated with a list of PlayerSubfactory objects. This list
 * corresponds to the list of player types that the factory can create.
 */
template <typename GeneratorT>
class PlayerSubfactory : public PlayerSubfactoryBase<typename GeneratorT::Game> {
 public:
  GeneratorT* create(GameServerBase* server) override { return new GeneratorT(server); }
};

/*
 * The PlayerFactory is a template class that facilitates the creation of player objects from
 * command line arguments.
 *
 * See cpp/connect4/PlayerFactory.hpp for an example of how to use this class.
 */
template <concepts::Game Game_>
class PlayerFactory {
 public:
  using Game = Game_;
  using Player = AbstractPlayer<Game>;
  using PlayerGenerator = AbstractPlayerGenerator<Game>;
  using PlayerSubfactoryBase = core::PlayerSubfactoryBase<Game>;

  struct PlayerGeneratorSeat {
    PlayerGenerator* generator = nullptr;
    int seat = -1;
  };
  using player_generator_seat_vec_t = std::vector<PlayerGeneratorSeat>;
  using player_generator_vec_t = std::vector<PlayerGenerator*>;
  using player_subfactory_vec_t = std::vector<PlayerSubfactoryBase*>;

  struct Params {
    auto make_options_description();

    std::string type;
    std::string name;
    std::string copy_from;
    int seat = -1;
  };

  /*
   * The constructor takes a vector of PlayerGenerator objects. Each PlayerGenerator provides a
   * recipe for how to determine whether a set of cmdline tokens match that generator, and if so,
   * how to create a player object.
   *
   * From this, the PlayerFactory generates the appropriate help message, with player-specific
   * details printed only if the associated type is passed via --player "--type=..."
   */
  PlayerFactory(const player_subfactory_vec_t& subfactories);

  void set_server(GameServerBase* server) { server_ = server; }
  player_generator_seat_vec_t parse(const std::vector<std::string>& player_strs);
  void print_help(const std::vector<std::string>& player_strs);

 private:
  static std::string type_str(const PlayerGenerator* generator);
  static bool matches(const PlayerGenerator* generator, const std::string& type);
  PlayerGenerator* parse_helper(const std::string& player_str, const std::string& name,
                                const std::vector<std::string>& tokens);

  player_subfactory_vec_t subfactories_;
  std::map<std::string, std::vector<std::string>> name_map_;
  GameServerBase* server_ = nullptr;
};

}  // namespace core

#include "inline/core/PlayerFactory.inl"
