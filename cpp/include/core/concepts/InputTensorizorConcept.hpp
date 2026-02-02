#pragma once

#include "core/concepts/KeysConcept.hpp"
#include "util/CppUtil.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

#include <concepts>

namespace core::concepts {

template <typename IT, typename Game>
concept InputTensorizor =
  requires(group::element_t sym, typename Game::State state, typename IT::StateIterator it) {
    typename IT::Tensor;
    typename IT::Keys;

    requires eigen_util::concepts::FTensor<typename IT::Tensor>;
    requires core::concepts::Keys<typename IT::Keys, Game>;

    // kNumStatesToEncode is the number of State's that are needed to tensorize a given state. If
    // the neural network does not need any previous State's, kNumStatesToEncode should be 1.
    { util::decay_copy(IT::kNumStatesToEncode) } -> std::same_as<int>;

    { IT::tensorize(sym) } -> std::same_as<typename IT::Tensor>;
    { IT::get_random_symmetry() } -> std::same_as<group::element_t>;
    { IT::update(state) } -> std::same_as<void>;
    { IT::undo(state) } -> std::same_as<void>;
    { IT::jump_to(it) } -> std::same_as<void>;
    { IT::clear() } -> std::same_as<void>;
  };

}  // namespace core::concepts
