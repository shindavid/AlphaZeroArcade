#pragma once

#include <core/TrainingTargets.hpp>

#include <concepts>

namespace core {
namespace concepts {

template <typename GTT, typename GameTypes>
concept GameTrainingTargets = requires {
  requires core::concepts::TrainingTargetList<typename GTT::List,
                                              typename GameTypes::GameLogView>;
};

}  // namespace concepts
}  // namespace core
