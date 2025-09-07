#pragma once

#include "core/concepts/Game.hpp"
#include "core/concepts/MctsConfigurationConcept.hpp"
#include "core/concepts/TrainingTargetsConcept.hpp"

namespace core::concepts {

template <typename ES>
concept EvalSpec = requires {
  requires core::concepts::Game<typename ES::Game>;
  requires core::concepts::TrainingTargets<typename ES::TrainingTargets, typename ES::Game>;
  requires core::concepts::MctsConfiguration<typename ES::MctsConfiguration>;
};

}  // namespace core::concepts
