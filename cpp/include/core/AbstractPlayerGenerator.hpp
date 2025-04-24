#pragma once

#include <ostream>
#include <string>
#include <vector>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <util/BoostUtil.hpp>

namespace core {

/*
 * An AbstractPlayerGenerator is a class that can create AbstractPlayer instances via its generate()
 * method. Fundamentally, you can think of it as a std::function<AbstractPlayer*()>, but with a few
 * extra features.
 *
 * This class exists because when simulating multiple games in parallel with GameServer, each
 * parallel thread needs to generate its own player. This is because each player has its own state,
 * meaning that multiple threads cannot share the same player. Because of this, GameServer must be
 * passed a player generator, rather than a player. If we never did parallel games, then this class
 * would not exist.
 *
 * Because the class exists, might as well add some extra features! Specifically, the class provides
 * a way to parse arguments passed from the command line. The command line will have one or more
 * instances of --player, looking something like:
 *
 * --player "--type=TUI --sub-arg1=val1 --sub-arg2 val2 ..."
 *
 * Each AbstractPlayerGenerator subclass will specify whether to match against a given --type=
 * string. It will also specify how to parse the other sub-arguments in order to construct a player
 * object.
 */
template <concepts::Game Game_>
class AbstractPlayerGenerator {
 public:
  using Game = Game_;

  virtual ~AbstractPlayerGenerator() = default;

  virtual std::string get_default_name() const = 0;

  /*
   * Returns a list of strings that match against the --type argument.
   *
   * We use a vector instead of a single string so that we can have multiple names for the same
   * player generator, thus allowing for shortcuts/aliases.
   */
  virtual std::vector<std::string> get_types() const = 0;

  /*
   * A short description of the player type, used in help messages.
   */
  virtual std::string get_description() const = 0;

  /*
   * Generate a new player. The caller is responsible for taking ownership of the pointer.
   *
   * game_slot_index designates a GameSlot (which might be local or remote) where the player
   * will be playing. Most subclasses can ignore this parameter. However, certain player types, as
   * an optimization, may wish to share data structures with other players sharing the same
   * game_slot_index. This parameter facilitates such sharing.
   */
  virtual AbstractPlayer<Game>* generate(game_slot_index_t game_slot_index) = 0;

  /*
   * Print help for this player generator, describing what parse_args() expects. This should
   * typically dispatch to a call to
   *
   * boost::program_options::options_description::printf(s)
   *
   * If there are no associated options for this player type, then this method does not need to be
   * overriden.
   */
  virtual void print_help(std::ostream& s) {}

  /*
   * Takes a list of arguments and parses them. This is called before generate(). The tokens that
   * will be passed here are extracted from the value of a --player argument.
   *
   * The --type, --name, and --seat parts of the tokens are removed first before passing to this
   * method.
   *
   * If there are no associated options for this player type, then this method does not need to be
   * overriden.
   */
  virtual void parse_args(const std::vector<std::string>& args) {}

  /*
   * Called when the GameServer starts. This allows the generator to do any initialization it needs
   * to do before it starts to construct players. This is useful for generators that need to
   * to do things like initialize object pools that will be shared among multiple players.
   */
  virtual void start_session() {}

  /*
   * Called when all games have been played. This is useful for when you want to report some
   * aggregate statistics over a series of games. Note that this functionality must exist here,
   * rather than at the player-level. This is because in the parallel case, two different games may
   * be played by different player instances, which don't know about each other.
   */
  virtual void end_session() {}

  /*
   * Some extra virtual functions that most subclasses can ignore.
   *
   * GameServer uses max_simultaneous_games() to determine how many games it can run in parallel.
   * The default return value of 0 indicates that the player can play an unlimited number of games
   * simultaneously. Currently, we only override this default for human players, which can only play
   * one game at a time due to interface limitations.
   */
  virtual int max_simultaneous_games() const { return 0; }

  const std::string& get_name() const { return name_; }

  /*
   * Validates name, raising an exception if the name is invalid (too long or uses invalid
   * characters).
   */
  void set_name(const std::string& name);

  /*
   * Convenience method that composes generate() with set_name().
   */
  AbstractPlayer<Game>* generate_with_name(game_slot_index_t);

 protected:
  /*
   * Helper function for parse_args() that some subclasses may find useful.
   */
  template <typename T>
  void parse_args_helper(T&& desc, const std::vector<std::string>& args) {
    namespace po2 = boost_util::program_options;
    po2::parse_args(desc, args);
  }

 private:
  std::string name_;
};

}  // namespace core

#include <inline/core/AbstractPlayerGenerator.inl>
