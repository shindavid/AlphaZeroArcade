#pragma once

#include "beta0/Player.hpp"
#include "beta0/PlayerGenerator.hpp"
#include "core/PlayerBundle.hpp"

namespace core {

template <>
struct PlayerBundle<kParadigmBetaZero> {
  template <typename Spec>
  using Player = ::beta0::Player<Spec>;

  template <typename PlayerT>
  using CompetitionPlayerGenerator = ::beta0::CompetitionPlayerGenerator<PlayerT>;

  template <typename PlayerT>
  using TrainingPlayerGenerator = ::beta0::TrainingPlayerGenerator<PlayerT>;

  template <typename GeneratorT>
  using Subfactory = ::beta0::Subfactory<GeneratorT>;
};

}  // namespace core
