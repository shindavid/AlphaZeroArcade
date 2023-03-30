#pragma once

#include <map>
#include <utility>

#include <connect4/C4GameState.hpp>

namespace c4 {

// TODO: finish implementation
class PlayerFactory {
public:
  using player_generator_t = std::function<Player*()>;

  /*
   * Creates a player generator out of args, and binds it to the given index. Attempts to double-register an index
   * will result in an exception.
   */
  static player_generator_t get_player_generator(int index, const char* args);

  /*
   * Returns a copy of the player generator bound to the given index. If no generator is bound to the given index,
   * an exception is thrown.
   */
  static player_generator_t copy_player_generator(int index);

private:
  static PlayerFactory* instance();

  using map_t = std::map<int, player_generator_t>;

  static PlayerFactory* instance_;
  map_t map_;
};

}  // namespace c4
