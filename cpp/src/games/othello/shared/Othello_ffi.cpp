#include <core/GameLog.hpp>
#include <core/FfiMacro.hpp>
#include <games/othello/Game.hpp>

using GameLog = core::GameLog<othello::Game>;
FFI_MACRO(GameLog);
