#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/PlayerResultConcept.hpp"
#include "util/EigenUtil.hpp"

#include <array>
#include <concepts>

namespace core::concepts {

template <class GRE, class Game>
concept GameResultEncoding =
  requires(const typename GRE::Tensor& const_t, typename GRE::Tensor& t,
           const std::array<typename GRE::PlayerResult, Game::Constants::kNumPlayers>& outcome,
           core::seat_index_t seat) {
    requires core::concepts::PlayerResult<typename GRE::PlayerResult>;
    requires eigen_util::concepts::FTensor<typename GRE::Tensor>;
    requires eigen_util::concepts::FArray<typename GRE::ValueArray>;

    { util::decay_copy(GRE::kMinValue) } -> std::same_as<float>;
    { util::decay_copy(GRE::kMaxValue) } -> std::same_as<float>;

    { GRE::encode(outcome) } -> std::same_as<typename GRE::Tensor>;
    { GRE::to_value_array(const_t) } -> std::same_as<typename GRE::ValueArray>;
    { GRE::left_rotate(t, seat) };
    { GRE::right_rotate(t, seat) };
  };

}  // namespace core::concepts
