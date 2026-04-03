#pragma once

#include "core/PlayerFactory.hpp"
#include "generic_players/alpha0/Player.hpp"
#include "generic_players/x0/PlayerGenerator.hpp"
#include "search/concepts/TraitsConcept.hpp"

#include <string>

namespace generic::alpha0 {

template <search::concepts::Traits Traits>
class CompetitionPlayerGenerator
    : public generic::x0::CompetitionPlayerGenerator<generic::alpha0::Player<Traits>> {
 public:
  using Base = generic::x0::CompetitionPlayerGenerator<generic::alpha0::Player<Traits>>;
  using Base::Base;

  std::string type_str() const override { return "alpha0-C"; }
  std::string get_description() const override { return "Competition alpha0 player"; }
};

template <search::concepts::Traits Traits>
class TrainingPlayerGenerator
    : public generic::x0::TrainingPlayerGenerator<generic::alpha0::Player<Traits>> {
 public:
  using Base = generic::x0::TrainingPlayerGenerator<generic::alpha0::Player<Traits>>;
  using Base::Base;

  std::string type_str() const override { return "alpha0-T"; }
  std::string get_description() const override { return "Training alpha0 player"; }
};

}  // namespace generic::alpha0

namespace core {

template <search::concepts::Traits Traits>
class PlayerSubfactory<generic::alpha0::CompetitionPlayerGenerator<Traits>>
    : public generic::x0::Subfactory<generic::alpha0::CompetitionPlayerGenerator<Traits>> {};

template <search::concepts::Traits Traits>
class PlayerSubfactory<generic::alpha0::TrainingPlayerGenerator<Traits>>
    : public generic::x0::Subfactory<generic::alpha0::TrainingPlayerGenerator<Traits>> {};

}  // namespace core
