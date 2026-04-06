#pragma once

#include "core/concepts/GameConcept.hpp"
#include "core/concepts/PolicyEncodingConcept.hpp"
#include "util/EigenUtil.hpp"

namespace core {

template <concepts::Game Game, concepts::PolicyEncoding PolicyEncoding>
struct ActionValueEncoding {
  using PolicyShape = PolicyEncoding::Shape;
  using Shape = eigen_util::extend_shape_t<PolicyShape, Game::Constants::kNumPlayers>;
  using Tensor = eigen_util::FTensor<Shape>;
};

}  // namespace core
