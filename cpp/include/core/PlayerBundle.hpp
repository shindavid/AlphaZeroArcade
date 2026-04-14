#pragma once

#include "core/SearchParadigm.hpp"

namespace core {

/*
 * Paradigm-keyed traits struct mapping SearchParadigm to Player, PlayerGenerator, and Subfactory
 * types.
 *
 * Each paradigm specializes this template to expose:
 *   - Player<Spec>
 *   - CompetitionPlayerGenerator<Player>
 *   - TrainingPlayerGenerator<Player>
 *   - Subfactory<GeneratorT>
 *
 * This allows game PlayerFactory code to construct the right generators from
 * Bindings::SupportedSpecs without directly depending on any paradigm namespace.
 */
template <SearchParadigm>
struct PlayerBundle;

}  // namespace core
