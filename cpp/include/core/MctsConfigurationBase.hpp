#pragma once

namespace core {
/*
 * This can be used as the base class of any Game::MctsConfiguration struct, in order to get default
 * values for some of the MCTS configuration parameters.
 */
struct MctsConfigurationBase {
  static constexpr bool kStoreStates = false;
};

}  // namespace core
