#pragma once

#include "util/CppUtil.hpp"

#include <magic_enum/magic_enum.hpp>
#include <magic_enum/magic_enum_format.hpp>

#include <cstdint>

namespace search {

enum Mode : int8_t { kCompetition, kTraining };

constexpr int kThreadWhitespaceLength = 50;  // for debug printing alignment
constexpr bool kEnableSearchDebug = IS_DEFINED(MCTS_DEBUG);

enum RootInitPurpose : int8_t { kForStandardSearch, kToLoadRootActionValues };

}  // namespace search
