#pragma once

#include "games/tictactoe/Game.hpp"
#include "generic_players/WebPlayer.hpp"

namespace tictactoe {

using WebPlayer = generic::WebPlayer<Game>;

}  // namespace tictactoe
