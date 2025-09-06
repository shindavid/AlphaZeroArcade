#pragma once

#include "core/concepts/Game.hpp"
#include "core/concepts/KeysConcept.hpp"
#include "core/concepts/TrainingTargetsConcept.hpp"

namespace core::concepts {

template <typename ES>
concept EvalSpec = requires {
  requires core::concepts::Game<typename ES::Game>;
  requires core::concepts::TrainingTargets<typename ES::TrainingTargets, typename ES::Game>;
  requires core::concepts::Keys<typename ES::Keys, typename ES::Game>;
};

}  // namespace core
