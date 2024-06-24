#include <core/GameLog.hpp>
#include <core/FfiMacro.hpp>
#include <games/connect4/Game.hpp>

using GameLog = core::GameLog<c4::Game>;
FFI_MACRO(GameLog);
