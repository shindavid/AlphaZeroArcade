#pragma once

#include "core/concepts/ParadigmSpecConcept.hpp"
#include "core/concepts/InputFrameConcept.hpp"
#include "core/concepts/MctsConfigurationConcept.hpp"
#include "core/concepts/NetworkHeadsConcept.hpp"
#include "core/concepts/SymmetriesConcept.hpp"
#include "core/concepts/TensorEncodingsConcept.hpp"
#include "core/concepts/TrainingTargetsConcept.hpp"
#include "core/concepts/TransposerConcept.hpp"

namespace alpha0::concepts {

template <typename ES>
concept Spec = core::concepts::ParadigmSpec<ES> && requires {
  requires core::concepts::TensorEncodings<typename ES::TensorEncodings>;
  requires core::concepts::InputFrame<typename ES::InputFrame, typename ES::Game::State>;
  requires core::concepts::Symmetries<typename ES::Symmetries, typename ES::Game,
                                      typename ES::TensorEncodings, typename ES::InputFrame>;
  requires core::concepts::Transposer<typename ES::Transposer, typename ES::Game::State>;
  requires core::concepts::TrainingTargets<typename ES::TrainingTargets, typename ES::Game>;
  requires core::concepts::NetworkHeads<typename ES::NetworkHeads, typename ES::Game>;
  requires core::concepts::MctsConfiguration<typename ES::MctsConfiguration>;
};

}  // namespace alpha0::concepts
