#pragma once

#include "alpha0/concepts/SpecConcept.hpp"

/*
 * This file contains metafunctions that create Spec types from other Spec types.
 */

namespace transforms {

/*
 * AddStateStorage is a game transformation that adds state storage to an Spec by setting
 * MctsConfiguration::kStoreStates to true.
 */
template <::alpha0::concepts::Spec Spec>
struct AddStateStorage : public Spec {
  struct MctsConfiguration : public Spec::MctsConfiguration {
    static constexpr bool kStoreStates = true;
  };
};

}  // namespace transforms
