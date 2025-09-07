#pragma once

#include "core/concepts/EvalSpecConcept.hpp"
#include "util/CppUtil.hpp"

namespace core {

/*
 * StableDataBase<EvalSpec, false> is an empty class that does nothing.
 *
 * StableDataBase<EvalSpec, true> is a class that stores a game state.
 *
 * StableData<EvalSpec> inherits from StableDataBase<EvalSpec, B>, with B set to true only if the
 * macro STORE_STATES is enabled OR if the EvalSpec's MctsConfiguration has kStoreStates set to true
 * (this is used in unit tests).
 *
 * This allows for us to store the game state in the node object, which can be useful for debugging
 * and analysis.
 *
 * Note that StableDataBase<EvalSpec, false> is an empty base-class, allowing us to get the empty
 * base-class optimization in StableData<EvalSpec>.
 */
template <core::concepts::EvalSpec EvalSpec, bool EnableStorage>
struct StableDataBase {
  using State = EvalSpec::Game::State;

  StableDataBase(const State&) {}
  const State* get_state() const { return nullptr; }
};

template <core::concepts::EvalSpec EvalSpec>
struct StableDataBase<EvalSpec, true> {
  using State = EvalSpec::Game::State;

  StableDataBase(const State& s) : state(s) {}
  const State* get_state() const { return &state; }

  State state;
};

template <core::concepts::EvalSpec EvalSpec>
constexpr bool kStoreStates = IS_DEFINED(STORE_STATES) || EvalSpec::MctsConfiguration::kStoreStates;

template <core::concepts::EvalSpec EvalSpec>
using StableDataBaseImpl = StableDataBase<EvalSpec, kStoreStates<EvalSpec>>;

}  // namespace core
