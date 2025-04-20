#pragma once

namespace core {

// Globals is a placeholder for global variables that we want to use across the codebase.
//
// The existence of this class is a bit of a hack, and we should try to avoid using it if possible.
struct Globals {
  // num_game_threads is the number of game threads that GameServer/GameServerProxy instantiates.
  // It is set in a specific place in the run() method of those classes.
  static int num_game_threads;  // From GameServer
};

}  // namespace core
