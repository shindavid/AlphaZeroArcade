#pragma once

#include <cstdint>

namespace search {

constexpr int kThreadWhitespaceLength = 50;  // for debug printing alignment

enum RootInitPurpose : int8_t { kForStandardSearch, kToLoadRootActionValues };

}  // namespace search
