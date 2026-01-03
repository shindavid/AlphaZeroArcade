#pragma once

#include "core/PlayerFactory.hpp"
#include "generic_players/beta0/Player.hpp"
#include "generic_players/x0/PlayerGenerator.hpp"
#include "search/concepts/TraitsConcept.hpp"

#include <string>
#include <vector>

namespace generic::beta0 {

template <search::concepts::Traits Traits>
class CompetitionPlayerGenerator
    : public generic::x0::CompetitionPlayerGenerator<generic::beta0::Player<Traits>> {
 public:
  using Base = generic::x0::CompetitionPlayerGenerator<generic::beta0::Player<Traits>>;
  using Base::Base;

  std::vector<std::string> get_types() const override {
    return {"beta0-C", "BetaZero-Competition"};
  }
  std::string get_description() const override { return "Competition BetaZero player"; }
};

template <search::concepts::Traits Traits>
class TrainingPlayerGenerator
    : public generic::x0::TrainingPlayerGenerator<generic::beta0::Player<Traits>> {
 public:
  using Base = generic::x0::TrainingPlayerGenerator<generic::beta0::Player<Traits>>;
  using Base::Base;

  std::vector<std::string> get_types() const override { return {"beta0-T", "BetaZero-Training"}; }
  std::string get_description() const override { return "Training BetaZero player"; }
};

}  // namespace generic::beta0

namespace core {

template <search::concepts::Traits Traits>
class PlayerSubfactory<generic::beta0::CompetitionPlayerGenerator<Traits>>
    : public generic::x0::Subfactory<generic::beta0::CompetitionPlayerGenerator<Traits>> {};

template <search::concepts::Traits Traits>
class PlayerSubfactory<generic::beta0::TrainingPlayerGenerator<Traits>>
    : public generic::x0::Subfactory<generic::beta0::TrainingPlayerGenerator<Traits>> {};

}  // namespace core
