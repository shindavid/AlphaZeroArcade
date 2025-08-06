#pragma once

#include "games/connect4/Game.hpp"
#include "generic_players/WebPlayer.hpp"

namespace c4 {

using WebPlayer = generic::WebPlayer<Game>;

}  // namespace c4
