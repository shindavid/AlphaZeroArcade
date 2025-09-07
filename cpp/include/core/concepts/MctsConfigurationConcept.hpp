#pragma once

#include "util/CppUtil.hpp"

#include <concepts>

namespace core {
namespace concepts {

template <class MC>
concept MctsConfiguration = requires {
  /*
   * kOpeningLength is a subjectively chosen value that indicates the number of moves that are
   * considered to be part of the "opening" phase of the game. This is used to select default
   * parameters for move-temperature and root-softmax-temperature. Those parameters will decay
   * using kOpeningLength as the *quarter-life* (or, 0.5 * kOpeningLength as the half-life).
   *
   * Roughly speaking, if you are willing to tolerate slightly suboptimal move-selection in the
   * first N moves of the game, tightening up to optimal play after that, then you should set
   * kOpeningLength to N. The reason to tolerate suboptimality in the opening is so that the agent
   * does not act too deterministically. This helps achieve data diversity in training, and also
   * more meaningful testing.
   *
   * KataGo effectively uses a value of 38 for this in 19x19 go.
   */
  { util::decay_copy(MC::kOpeningLength) } -> std::same_as<float>;

  /*
   * If kStoreStates is enabled, then the Node will store the State in the Node itself. This can
   * also be enabled by enabling the STORE_STATES macro.
   *
   * If using the core::MctsConfigurationBase base-class, this will be false by default.
   */
  { util::decay_copy(MC::kStoreStates) } -> std::same_as<bool>;
};

}  // namespace concepts
}  // namespace core
