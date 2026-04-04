#pragma once

#include <concepts>
#include <random>

namespace core {

namespace concepts {

/*
 * All Game::MoveList types ML must satisfy core::concepts::MoveList<ML, Move>.
 *
 * Requirements:
 * - size()     -> convertible to int
 * - empty()    -> convertible to bool
 * - get_random(std::mt19937&) -> Move
 * - add(const Move&)
 * - contains(const Move&) -> convertible to bool
 * - begin() / end()
 */
template <class ML, class Move>
concept MoveList =
  requires(ML ml, const ML& cml, const Move& m, std::mt19937& prng) {
    { cml.size() } -> std::convertible_to<int>;
    { cml.empty() } -> std::convertible_to<bool>;
    { cml.get_random(prng) } -> std::same_as<Move>;
    { ml.add(m) };
    { cml.contains(m) } -> std::convertible_to<bool>;
    { cml.begin() };
    { cml.end() };
  };

}  // namespace concepts

}  // namespace core
