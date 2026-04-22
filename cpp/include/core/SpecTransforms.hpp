#pragma once

/*
 * This file contains metafunctions that create Spec types from other Spec types.
 */

namespace transforms {

/*
 * AddStateStorage is a game transformation that adds state storage to an Spec by setting
 * MctsConfiguration::kStoreStates to true.
 */
template <typename Spec>
struct AddStateStorage : public Spec {
  struct MctsConfiguration : public Spec::MctsConfiguration {
    static constexpr bool kStoreStates = true;
  };
};

}  // namespace transforms
