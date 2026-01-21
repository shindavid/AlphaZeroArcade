#pragma once

#include "core/Constants.hpp"
#include "core/concepts/GameConcept.hpp"
#include "core/concepts/MctsConfigurationConcept.hpp"
#include "core/concepts/NetworkHeadsConcept.hpp"
#include "core/concepts/TrainingTargetsConcept.hpp"
#include "util/CppUtil.hpp"

#include <concepts>

namespace core::concepts {

template <typename ES>
concept EvalSpec = requires {
  // The name of the game. Should match GameSpec.name in python.
  { util::decay_copy(ES::kParadigm) } -> std::same_as<core::SearchParadigm>;

  requires core::concepts::Game<typename ES::Game>;
  requires core::concepts::TrainingTargets<typename ES::TrainingTargets, typename ES::Game>;
  requires core::concepts::NetworkHeads<typename ES::NetworkHeads, typename ES::Game>;
  requires core::concepts::MctsConfiguration<typename ES::MctsConfiguration>;
};

}  // namespace core::concepts
