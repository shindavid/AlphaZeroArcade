#pragma once

#include "core/Constants.hpp"
#include "core/concepts/GameConcept.hpp"
#include "core/concepts/InputFrameConcept.hpp"
#include "core/concepts/InputTensorizorConcept.hpp"
#include "core/concepts/MctsConfigurationConcept.hpp"
#include "core/concepts/NetworkHeadsConcept.hpp"
#include "core/concepts/SymmetriesConcept.hpp"
#include "core/concepts/TrainingTargetsConcept.hpp"
#include "core/concepts/TransposerConcept.hpp"
#include "util/CppUtil.hpp"

#include <concepts>

namespace core::concepts {

template <typename ES>
concept EvalSpec = requires {
  // The name of the game. Should match GameSpec.name in python.
  { util::decay_copy(ES::kParadigm) } -> std::same_as<core::SearchParadigm>;

  requires core::concepts::Game<typename ES::Game>;
  requires core::concepts::InputFrame<typename ES::InputFrame, typename ES::Game::State>;
  requires core::concepts::Symmetries<typename ES::Symmetries, typename ES::Game,
                                      typename ES::InputFrame>;
  requires core::concepts::Transposer<typename ES::Transposer, typename ES::Game::State>;
  requires core::concepts::InputTensorizor<typename ES::InputTensorizor, typename ES::Game::State,
                                           typename ES::InputFrame>;
  requires core::concepts::TrainingTargets<typename ES::TrainingTargets, typename ES::Game>;
  requires core::concepts::NetworkHeads<typename ES::NetworkHeads, typename ES::Game>;
  requires core::concepts::MctsConfiguration<typename ES::MctsConfiguration>;
};

}  // namespace core::concepts
