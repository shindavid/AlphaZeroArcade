#pragma once

#include "games/hex/Game.hpp"
#include "generic_players/WebPlayer.hpp"

namespace hex {

// TODO: add hex-specific overrides if needed.
class WebPlayer : public generic::WebPlayer<Game> {};

}  // namespace hex
