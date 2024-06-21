#include <core/GameLog.hpp>
#include <core/FfiMacro.hpp>
#include <games/tictactoe/Game.hpp>

using GameLog = core::GameLog<tictactoe::Game>;
FFI_MACRO(GameLog);
