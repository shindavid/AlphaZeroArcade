#pragma once

#include "core/SimplePolicyEncoding.hpp"
#include "games/tictactoe/Game.hpp"
#include "games/tictactoe/InputFrame.hpp"

namespace tictactoe {

using PolicyEncoding = core::SimplePolicyEncoding<Game, InputFrame>;

}  // namespace tictactoe
