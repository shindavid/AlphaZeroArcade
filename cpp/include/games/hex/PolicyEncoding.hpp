#pragma once

#include "core/SimplePolicyEncoding.hpp"
#include "games/hex/Game.hpp"
#include "games/hex/InputFrame.hpp"

namespace hex {

using PolicyEncoding = core::SimplePolicyEncoding<Game, InputFrame>;

}  // namespace hex
