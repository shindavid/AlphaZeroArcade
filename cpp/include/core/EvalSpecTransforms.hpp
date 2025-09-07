#pragma once

#include "core/concepts/EvalSpecConcept.hpp"

/*
 * This file contains metafunctions that create EvalSpec types from other EvalSpec types.
 */

namespace transforms {

/*
 * AddStateStorage is a game transformation that adds state storage to an EvalSpec by setting
 * MctsConfiguration::kStoreStates to true.
 */
template <core::concepts::EvalSpec EvalSpec>
struct AddStateStorage : public EvalSpec {
  struct MctsConfiguration : public EvalSpec::MctsConfiguration {
    static constexpr bool kStoreStates = true;
  };
};

}  // namespace transforms
