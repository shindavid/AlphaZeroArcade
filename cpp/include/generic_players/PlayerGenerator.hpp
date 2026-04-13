#pragma once

#include "alpha0/Player.hpp"
#include "alpha0/PlayerGenerator.hpp"
#include "alpha0/concepts/SpecConcept.hpp"
#include "core/PlayerFactory.hpp"

namespace generic {

// Thin wrappers that map Spec → alpha0::Player<Spec> and forward to the alpha0 generators.
// These exist to provide a Spec-based interface for PlayerFactory subfactory specializations.
template <::alpha0::concepts::Spec Spec>
class CompetitionPlayerGenerator : public alpha0::CompetitionPlayerGenerator<alpha0::Player<Spec>> {
 public:
  using Base = alpha0::CompetitionPlayerGenerator<alpha0::Player<Spec>>;
  using Base::Base;
};

template <::alpha0::concepts::Spec Spec>
class TrainingPlayerGenerator : public alpha0::TrainingPlayerGenerator<alpha0::Player<Spec>> {
 public:
  using Base = alpha0::TrainingPlayerGenerator<alpha0::Player<Spec>>;
  using Base::Base;
};

}  // namespace generic

namespace core {

template <::alpha0::concepts::Spec Spec>
class PlayerSubfactory<::generic::CompetitionPlayerGenerator<Spec>>
    : public ::alpha0::Subfactory<::generic::CompetitionPlayerGenerator<Spec>> {};

template <::alpha0::concepts::Spec Spec>
class PlayerSubfactory<::generic::TrainingPlayerGenerator<Spec>>
    : public ::alpha0::Subfactory<::generic::TrainingPlayerGenerator<Spec>> {};

}  // namespace core
