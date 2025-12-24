#pragma once

#include "core/PlayerFactory.hpp"
#include "generic_players/alpha0/Player.hpp"
#include "generic_players/x0/PlayerGenerator.hpp"
#include "search/concepts/TraitsConcept.hpp"

#include <string>
#include <vector>

namespace generic::alpha0 {

template <search::concepts::Traits Traits>
class CompetitionPlayerGenerator
    : public generic::x0::CompetitionPlayerGenerator<generic::alpha0::Player<Traits>> {
 public:
  using Base = generic::x0::CompetitionPlayerGenerator<generic::alpha0::Player<Traits>>;
  using Base::Base;

  std::vector<std::string> get_types() const override {
    return {"alpha0-C", "AlphaZero-Competition", "MCTS-C"};  // We keep MCTS-C for nostalgic reasons
  }
  std::string get_description() const override {
    return "Competition AlphaZero player";
  }
};

template <search::concepts::Traits Traits>
class TrainingPlayerGenerator
    : public generic::x0::TrainingPlayerGenerator<generic::alpha0::Player<Traits>> {
 public:
  using Base = generic::x0::TrainingPlayerGenerator<generic::alpha0::Player<Traits>>;
  using Base::Base;

  std::vector<std::string> get_types() const override {
    return {"alpha0-T", "AlphaZero-Training", "MCTS-T"};  // We keep MCTS-T for nostalgic reasons
  }
  std::string get_description() const override {
    return "Training AlphaZero player";
  }
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
