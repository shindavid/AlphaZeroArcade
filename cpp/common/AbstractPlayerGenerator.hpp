#pragma once

#include <ostream>
#include <string>
#include <vector>

#include <common/AbstractPlayer.hpp>
#include <common/GameStateConcept.hpp>

namespace common {

/*
 * An AbstractPlayerGenerator is a class that can create AbstractPlayer instances via its generate() method.
 *
 * This class exists because when simulating multiple games in parallel with GameServer, each parallel thread needs to
 * generate its own player. This is because each player has its own state, meaning that multiple threads cannot share
 * the same player. Because of this, GameServer must accept a player generator, rather than a player.
 *
 * Additionally, this class provides a way to parse arguments passed from the command line. The command line will have
 * something like:
 *
 * --player "--type=TUI --arg1=val1 --arg2 val2"
 *
 * For a given AbstractPlayerGenerator instance X, if X.get_types() includes "TUI", then X is considered a match for
 * the given string. The string "--arg1=val1 --arg2 val2" is then split into tokens and passed to X.parse_args(). The
 * generate() method can then be invoked one or more times to create player intsances.
 */
template<GameStateConcept GameState>
class AbstractPlayerGenerator {
public:
  virtual ~AbstractPlayerGenerator() = default;

  /*
   * Returns a list of strings that match against the --type argument.
   *
   * We use a vector instead of a single string so that we can have multiple names for the same player generator.
   */
  virtual std::vector<std::string> get_types() const = 0;

  /*
   * A short description of the player type, used in help messages.
   */
  virtual std::string get_description() const = 0;

  /*
   * Generate a new player. The caller is responsible for deleting the player.
   *
   * play_address is the address where the player will be playing in. Most subclasses can ignore this parameter.
   * However, certain player types, as an optimization, may wish to share data structures with other players at the
   * same address. This parameter facilitates such sharing.
   */
  virtual AbstractPlayer<GameState>* generate(void* play_address) = 0;

  /*
   * Print help for this player generator, describing what parse_args() expects. This should typically dispatch to a
   * call to
   *
   * boost::program_options::options_description::printf(s)
   *
   * If there are no associated options for this player type, then this method does not need to be overriden.
   */
  virtual void print_help(std::ostream& s) {}

  /*
   * Takes a list of arguments and parses them. This is called before generate().
   *
   * Note that the "--type=" part of the command line string is removed from args.
   *
   * If there are no associated options for this player type, then this method does not need to be overriden.
   */
  virtual void parse_args(const std::vector<std::string>& args) {}
};

}  // namespace common
