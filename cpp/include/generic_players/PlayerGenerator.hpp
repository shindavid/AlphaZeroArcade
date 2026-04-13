#pragma once

#include "core/PlayerFactory.hpp"
#include "core/SearchParadigm.hpp"
#include "generic_players/alpha0/Player.hpp"
#include "generic_players/alpha0/PlayerGenerator.hpp"
#include "search/concepts/SearchSpecConcept.hpp"

#include <format>
#include <string>

namespace generic {

// Selects the correct Player type for a given SearchSpec based on its paradigm.
// Primary template is intentionally left undefined to give a hard error for unhandled paradigms.
template <search::concepts::SearchSpec SearchSpec,
          core::SearchParadigm = SearchSpec::EvalSpec::kParadigm>
struct PlayerFor;

template <search::concepts::SearchSpec SearchSpec>
struct PlayerFor<SearchSpec, core::kParadigmAlphaZero> {
  using type = generic::alpha0::Player<SearchSpec>;
};

template <search::concepts::SearchSpec SearchSpec>
using PlayerFor_t = typename PlayerFor<SearchSpec>::type;

// Unified CompetitionPlayerGenerator: dispatches to the correct Player type automatically.
template <search::concepts::SearchSpec SearchSpec>
class CompetitionPlayerGenerator
    : public generic::alpha0::CompetitionPlayerGenerator<PlayerFor_t<SearchSpec>> {
 public:
  using Base = generic::alpha0::CompetitionPlayerGenerator<PlayerFor_t<SearchSpec>>;
  using Base::Base;
  using SearchParadigmTraits = core::SearchParadigmTraits<SearchSpec::EvalSpec::kParadigm>;

  std::string type_str() const override { return std::format("{}-C", SearchParadigmTraits::kName); }
  std::string get_description() const override {
    return std::format("Competition {} player", SearchParadigmTraits::kName);
  }
};

// Unified TrainingPlayerGenerator: dispatches to the correct Player type automatically.
template <search::concepts::SearchSpec SearchSpec>
class TrainingPlayerGenerator
    : public generic::alpha0::TrainingPlayerGenerator<PlayerFor_t<SearchSpec>> {
 public:
  using Base = generic::alpha0::TrainingPlayerGenerator<PlayerFor_t<SearchSpec>>;
  using Base::Base;
  using SearchParadigmTraits = core::SearchParadigmTraits<SearchSpec::EvalSpec::kParadigm>;

  std::string type_str() const override { return std::format("{}-T", SearchParadigmTraits::kName); }
  std::string get_description() const override {
    return std::format("Training {} player", SearchParadigmTraits::kName);
  }
};

}  // namespace generic

namespace core {

template <search::concepts::SearchSpec SearchSpec>
class PlayerSubfactory<generic::CompetitionPlayerGenerator<SearchSpec>>
    : public generic::alpha0::Subfactory<generic::CompetitionPlayerGenerator<SearchSpec>> {};

template <search::concepts::SearchSpec SearchSpec>
class PlayerSubfactory<generic::TrainingPlayerGenerator<SearchSpec>>
    : public generic::alpha0::Subfactory<generic::TrainingPlayerGenerator<SearchSpec>> {};

}  // namespace core
