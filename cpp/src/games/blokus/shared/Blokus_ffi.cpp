#include <core/GameLog.hpp>
#include <core/FfiMacro.hpp>
#include <games/blokus/Game.hpp>

using GameLog = core::GameLog<blokus::Game>;
FFI_MACRO(GameLog);
