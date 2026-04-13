#pragma once

#include "core/PlayerFactory.hpp"
#include "generic_players/alpha0/Player.hpp"
#include "generic_players/alpha0/PlayerGenerator.hpp"
#include "search/concepts/SpecConcept.hpp"

namespace generic {

// Thin wrappers that map Spec → alpha0::Player<Spec> and forward to the alpha0 generators.
// These exist to provide a Spec-based interface for PlayerFactory subfactory specializations.
template <search::concepts::Spec Spec>
class CompetitionPlayerGenerator
    : public generic::alpha0::CompetitionPlayerGenerator<generic::alpha0::Player<Spec>> {
 public:
  using Base = generic::alpha0::CompetitionPlayerGenerator<generic::alpha0::Player<Spec>>;
  using Base::Base;
};

template <search::concepts::Spec Spec>
class TrainingPlayerGenerator
    : public generic::alpha0::TrainingPlayerGenerator<generic::alpha0::Player<Spec>> {
 public:
  using Base = generic::alpha0::TrainingPlayerGenerator<generic::alpha0::Player<Spec>>;
  using Base::Base;
};

}  // namespace generic

namespace core {

template <search::concepts::Spec Spec>
class PlayerSubfactory<generic::CompetitionPlayerGenerator<Spec>>
    : public generic::alpha0::Subfactory<generic::CompetitionPlayerGenerator<Spec>> {};

template <search::concepts::Spec Spec>
class PlayerSubfactory<generic::TrainingPlayerGenerator<Spec>>
    : public generic::alpha0::Subfactory<generic::TrainingPlayerGenerator<Spec>> {};

}  // namespace core
