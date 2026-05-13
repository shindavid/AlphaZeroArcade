#pragma once

#include "core/concepts/InputFrameConcept.hpp"
#include "core/concepts/MctsConfigurationConcept.hpp"
#include "core/concepts/NetworkHeadsConcept.hpp"
#include "core/concepts/ParadigmSpecConcept.hpp"
#include "core/concepts/SymmetriesConcept.hpp"
#include "core/concepts/TensorEncodingsConcept.hpp"
#include "core/concepts/TrainingTargetsConcept.hpp"
#include "core/concepts/TransposerConcept.hpp"
#include "util/CppUtil.hpp"

namespace beta0::concepts {

// Dimensions for the BetaZero CPU-side BackupNet (must match Python spec).
template <typename D>
concept BackupNetDims = requires {
  { util::decay_copy(D::kStaticLatentDim) } -> std::same_as<int>;
  { util::decay_copy(D::kEmbedDim) } -> std::same_as<int>;
  { util::decay_copy(D::kBackupLayer1Dim) } -> std::same_as<int>;
  { util::decay_copy(D::kBackupLayer2Dim) } -> std::same_as<int>;
  { util::decay_copy(D::kActionLatentDim) } -> std::same_as<int>;
};

template <typename ES>
concept Spec = core::concepts::ParadigmSpec<ES> && requires {
  requires core::concepts::TensorEncodings<typename ES::TensorEncodings>;
  requires core::concepts::InputFrame<typename ES::InputFrame, typename ES::Game::State>;
  requires core::concepts::Symmetries<typename ES::Symmetries, typename ES::Game,
                                      typename ES::TensorEncodings, typename ES::InputFrame>;
  requires core::concepts::Transposer<typename ES::Transposer, typename ES::Game::State>;
  requires core::concepts::TrainingTargets<typename ES::TrainingTargets, typename ES::Game>;
  requires core::concepts::NetworkHeads<typename ES::NetworkHeads>;
  requires core::concepts::MctsConfiguration<typename ES::MctsConfiguration>;
  requires BackupNetDims<typename ES::BackupNetDims>;
};

}  // namespace beta0::concepts
