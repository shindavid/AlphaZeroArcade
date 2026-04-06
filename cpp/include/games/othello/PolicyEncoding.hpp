#pragma once

#include "core/SimplePolicyEncoding.hpp"
#include "games/othello/Game.hpp"
#include "games/othello/InputFrame.hpp"

namespace othello {

using PolicyEncoding = core::SimplePolicyEncoding<Game, InputFrame>;

}  // namespace othello
