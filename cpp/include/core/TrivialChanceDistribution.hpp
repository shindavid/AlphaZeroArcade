#pragma once

#include "util/Exceptions.hpp"

#include <random>

namespace core {

// This is never actually called, it's just a default used in RulesBase for games that don't have
// chance events. It has the right interface to be used as a ChanceDistribution, but will throw an
// exception if you try to use it.
template <typename Move>
struct TrivialChanceDistribution {
  Move sample(std::mt19937& prng) const {
    throw util::Exception("Chance distribution not implemented for this game");
  }

  float get(const Move&) const {
    throw util::Exception("Chance distribution not implemented for this game");
  }
};
}  // namespace core
