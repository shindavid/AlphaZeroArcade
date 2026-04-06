#pragma once

#include "core/SimplePolicyEncoding.hpp"
#include "games/connect4/Game.hpp"
#include "games/connect4/InputFrame.hpp"

namespace c4 {

using PolicyEncoding = core::SimplePolicyEncoding<Game, InputFrame>;

}  // namespace c4
