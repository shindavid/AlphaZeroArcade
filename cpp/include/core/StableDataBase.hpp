#pragma once

#include "alpha0/concepts/SpecConcept.hpp"
#include "util/CppUtil.hpp"

namespace core {

/*
 * StableDataBase<Spec, false> is an empty class that does nothing.
 *
 * StableDataBase<Spec, true> is a class that stores a game state.
 *
 * StableData<Spec> inherits from StableDataBase<Spec, B>, with B set to true only if the
 * macro STORE_STATES is enabled OR if the Spec's MctsConfiguration has kStoreStates set to true
 * (this is used in unit tests).
 *
 * This allows for us to store the game state in the node object, which can be useful for debugging
 * and analysis.
 *
 * Note that StableDataBase<Spec, false> is an empty base-class, allowing us to get the empty
 * base-class optimization in StableData<Spec>.
 */
template <::alpha0::concepts::Spec Spec, bool EnableStorage>
struct StableDataBase {
  using State = Spec::Game::State;

  StableDataBase(const State&) {}
  const State* get_state() const { return nullptr; }
};

template <::alpha0::concepts::Spec Spec>
struct StableDataBase<Spec, true> {
  using State = Spec::Game::State;

  StableDataBase(const State& s) : state(s) {}
  const State* get_state() const { return &state; }

  State state;
};

template <::alpha0::concepts::Spec Spec>
constexpr bool kStoreStates = IS_DEFINED(STORE_STATES) || Spec::MctsConfiguration::kStoreStates;

template <::alpha0::concepts::Spec Spec>
using StableDataBaseImpl = StableDataBase<Spec, kStoreStates<Spec>>;

}  // namespace core
