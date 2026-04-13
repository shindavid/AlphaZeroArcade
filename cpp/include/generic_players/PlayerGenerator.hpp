#pragma once

#include "core/PlayerFactory.hpp"
#include "core/SearchParadigm.hpp"
#include "generic_players/alpha0/Player.hpp"
#include "generic_players/alpha0/PlayerGenerator.hpp"
#include "search/concepts/SpecConcept.hpp"

#include <format>
#include <string>

namespace generic {

// Selects the correct Player type for a given Spec based on its paradigm.
// Primary template is intentionally left undefined to give a hard error for unhandled paradigms.
template <search::concepts::Spec Spec,
          core::SearchParadigm = Spec::EvalSpec::kParadigm>
struct PlayerFor;

template <search::concepts::Spec Spec>
struct PlayerFor<Spec, core::kParadigmAlphaZero> {
  using type = generic::alpha0::Player<Spec>;
};

template <search::concepts::Spec Spec>
using PlayerFor_t = typename PlayerFor<Spec>::type;

// Unified CompetitionPlayerGenerator: dispatches to the correct Player type automatically.
template <search::concepts::Spec Spec>
class CompetitionPlayerGenerator
    : public generic::alpha0::CompetitionPlayerGenerator<PlayerFor_t<Spec>> {
 public:
  using Base = generic::alpha0::CompetitionPlayerGenerator<PlayerFor_t<Spec>>;
  using Base::Base;
  using SearchParadigmTraits = core::SearchParadigmTraits<Spec::EvalSpec::kParadigm>;

  std::string type_str() const override { return std::format("{}-C", SearchParadigmTraits::kName); }
  std::string get_description() const override {
    return std::format("Competition {} player", SearchParadigmTraits::kName);
  }
};

// Unified TrainingPlayerGenerator: dispatches to the correct Player type automatically.
template <search::concepts::Spec Spec>
class TrainingPlayerGenerator
    : public generic::alpha0::TrainingPlayerGenerator<PlayerFor_t<Spec>> {
 public:
  using Base = generic::alpha0::TrainingPlayerGenerator<PlayerFor_t<Spec>>;
  using Base::Base;
  using SearchParadigmTraits = core::SearchParadigmTraits<Spec::EvalSpec::kParadigm>;

  std::string type_str() const override { return std::format("{}-T", SearchParadigmTraits::kName); }
  std::string get_description() const override {
    return std::format("Training {} player", SearchParadigmTraits::kName);
  }
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
