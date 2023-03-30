#pragma once

#include <map>
#include <utility>

#include <connect4/C4GameState.hpp>

namespace c4 {

// TODO: finish implementation
class PlayerFactory {
public:
  using player_generator_t = std::function<Player*()>;

  static player_generator_t get_player_generator(int player_index, const char* args);
  static player_generator_t copy_player_generator(int player_index);

private:
  static PlayerFactory* instance();

  using map_t = std::map<int, player_generator_t>;

  static PlayerFactory* instance_;
  map_t map_;
};

}  // namespace c4
