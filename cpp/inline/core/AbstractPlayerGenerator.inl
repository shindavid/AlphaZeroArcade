#include <core/AbstractPlayerGenerator.hpp>

#include <cctype>

#include <core/Constants.hpp>
#include <util/Exception.hpp>

namespace core {

template <concepts::Game Game>
void AbstractPlayerGenerator<Game>::set_name(const std::string& name) {
  // check that only alphanumeric, dash or underscore are used in name:
  for (char c : name) {
    if (!std::isalnum(c) && c != '-' && c != '_') {
      throw util::CleanException("Invalid character in player name (\"%s\")", name.c_str());
    }
  }

  int name_size = name.size();
  if (name_size > kMaxNameLength) {
    throw util::CleanException("Player name (\"%s\") too long (%d > %d)", name.c_str(), name_size,
                               kMaxNameLength);
  }

  name_ = name;
}

template <concepts::Game Game>
AbstractPlayer<Game>* AbstractPlayerGenerator<Game>::generate_with_name(
    game_thread_id_t game_thread_id) {
  auto player = generate(game_thread_id);
  player->set_name(name_);
  return player;
}

}  // namespace core
