#include "core/AbstractPlayerGenerator.hpp"

#include "core/Constants.hpp"
#include "util/Exceptions.hpp"

#include <cctype>

namespace core {

template <concepts::Game Game>
void AbstractPlayerGenerator<Game>::set_name(const std::string& name) {
  // check that only alphanumeric, dash or underscore are used in name:
  for (char c : name) {
    if (!std::isalnum(c) && c != '-' && c != '_') {
      throw util::CleanException("Invalid character in player name (\"{}\")", name);
    }
  }

  int name_size = name.size();
  if (name_size > kMaxNameLength) {
    throw util::CleanException("Player name (\"{}\") too long ({} > {})", name, name_size,
                               kMaxNameLength);
  }

  name_ = name;
}

template <concepts::Game Game>
AbstractPlayer<Game>* AbstractPlayerGenerator<Game>::generate_with_name(
  game_slot_index_t game_slot_index) {
  auto player = generate(game_slot_index);
  player->set_name(name_);
  return player;
}

}  // namespace core
