#pragma once

#include "core/SearchParadigm.hpp"

namespace core {

/*
 * Paradigm-keyed traits struct mapping SearchParadigm to GameReadLog and GameWriteLog types.
 *
 * Each paradigm (e.g., alpha0) specializes this template to declare its GameReadLog and
 * GameWriteLog template aliases. This allows code in search/ to resolve game log types from a Spec
 * without directly depending on any paradigm namespace.
 */
template <SearchParadigm>
struct GameLogBundle;

// Convenience aliases: resolve GameReadLog / GameWriteLog for a given Spec via its kParadigm.
template <typename Spec>
using GameReadLogFor_t = typename GameLogBundle<Spec::kParadigm>::template GameReadLog<Spec>;

template <typename Spec>
using GameWriteLogFor_t = typename GameLogBundle<Spec::kParadigm>::template GameWriteLog<Spec>;

}  // namespace core
