#pragma once

#include "util/CppUtil.hpp"

#include <concepts>
#include <random>

namespace core {

namespace concepts {

/*
 * All Game::MoveSet types MS must satisfy core::concepts::MoveSet<MS, Move>.
 *
 * Requirements:
 * - size()     -> convertible to int
 * - empty()    -> convertible to bool
 * - get_random(std::mt19937&) -> Move
 * - add(const Move&)
 * - contains(const Move&) -> convertible to bool
 * - begin() / end()
 */
template <class MS, class Move>
concept MoveSet = requires(MS ms, const MS& cms, const Move& m, std::mt19937& prng) {
  { util::decay_copy(MS::kSortedByMove) } -> std::same_as<bool>;
  { cms.size() } -> std::convertible_to<int>;
  { cms.empty() } -> std::same_as<bool>;
  { cms.get_random(prng) } -> std::same_as<Move>;
  { ms.add(m) };
  { cms.contains(m) } -> std::same_as<bool>;
  { cms.begin() };
  { cms.end() };
};

}  // namespace concepts

}  // namespace core
