#pragma once

namespace core {

// This can be used as the base class of any Game::Constants struct, in order to get default
// values for some of the constants.
struct ConstantsBase {
  static constexpr int kNumPreviousStatesToEncode = 0;
};

}  // namespace core
