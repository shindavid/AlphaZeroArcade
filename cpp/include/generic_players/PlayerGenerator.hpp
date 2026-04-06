#pragma once

#include "core/PlayerFactory.hpp"
#include "core/SearchParadigm.hpp"
#include "generic_players/alpha0/Player.hpp"
#include "generic_players/beta0/Player.hpp"
#include "generic_players/x0/PlayerGenerator.hpp"
#include "search/concepts/TraitsConcept.hpp"

#include <format>
#include <string>

namespace generic {

// Selects the correct Player type for a given Traits based on its paradigm.
// Primary template is intentionally left undefined to give a hard error for unhandled paradigms.
template <search::concepts::Traits Traits, core::SearchParadigm = Traits::EvalSpec::kParadigm>
struct PlayerFor;

template <search::concepts::Traits Traits>
struct PlayerFor<Traits, core::kParadigmAlphaZero> {
  using type = generic::alpha0::Player<Traits>;
};

template <search::concepts::Traits Traits>
struct PlayerFor<Traits, core::kParadigmBetaZero> {
  using type = generic::beta0::Player<Traits>;
};

template <search::concepts::Traits Traits>
using PlayerFor_t = typename PlayerFor<Traits>::type;

// Unified CompetitionPlayerGenerator: dispatches to the correct Player type automatically.
template <search::concepts::Traits Traits>
class CompetitionPlayerGenerator
    : public generic::x0::CompetitionPlayerGenerator<PlayerFor_t<Traits>> {
 public:
  using Base = generic::x0::CompetitionPlayerGenerator<PlayerFor_t<Traits>>;
  using Base::Base;
  using SearchParadigmTraits = core::SearchParadigmTraits<Traits::EvalSpec::kParadigm>;

  std::string type_str() const override { return std::format("{}-C", SearchParadigmTraits::kName); }
  std::string get_description() const override {
    return std::format("Competition {} player", SearchParadigmTraits::kName);
  }
};

// Unified TrainingPlayerGenerator: dispatches to the correct Player type automatically.
template <search::concepts::Traits Traits>
class TrainingPlayerGenerator : public generic::x0::TrainingPlayerGenerator<PlayerFor_t<Traits>> {
 public:
  using Base = generic::x0::TrainingPlayerGenerator<PlayerFor_t<Traits>>;
  using Base::Base;
  using SearchParadigmTraits = core::SearchParadigmTraits<Traits::EvalSpec::kParadigm>;

  std::string type_str() const override { return std::format("{}-T", SearchParadigmTraits::kName); }
  std::string get_description() const override {
    return std::format("Training {} player", SearchParadigmTraits::kName);
  }
};

}  // namespace generic

namespace core {

template <search::concepts::Traits Traits>
class PlayerSubfactory<generic::CompetitionPlayerGenerator<Traits>>
    : public generic::x0::Subfactory<generic::CompetitionPlayerGenerator<Traits>> {};

template <search::concepts::Traits Traits>
class PlayerSubfactory<generic::TrainingPlayerGenerator<Traits>>
    : public generic::x0::Subfactory<generic::TrainingPlayerGenerator<Traits>> {};

}  // namespace core
