#pragma once

#include "core/SimplePolicyEncoding.hpp"
#include "games/nim/Game.hpp"
#include "games/nim/InputFrame.hpp"

namespace nim {

using PolicyEncoding = core::SimplePolicyEncoding<Game, InputFrame>;

}  // namespace nim
