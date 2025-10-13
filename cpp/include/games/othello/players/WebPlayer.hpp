#pragma once

#include "games/othello/Game.hpp"
#include "generic_players/WebPlayer.hpp"

namespace othello {

using WebPlayer = generic::WebPlayer<Game>;

}  // namespace othello
