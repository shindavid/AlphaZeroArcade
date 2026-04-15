#pragma once

#include "alpha0/Player.hpp"
#include "alpha0/PlayerGenerator.hpp"
#include "core/PlayerBundle.hpp"

namespace core {

template <>
struct PlayerBundle<kParadigmAlphaZero> {
  template <typename Spec>
  using Player = ::alpha0::Player<Spec>;

  template <typename PlayerT>
  using CompetitionPlayerGenerator = ::alpha0::CompetitionPlayerGenerator<PlayerT>;

  template <typename PlayerT>
  using TrainingPlayerGenerator = ::alpha0::TrainingPlayerGenerator<PlayerT>;

  template <typename GeneratorT>
  using Subfactory = ::alpha0::Subfactory<GeneratorT>;
};

}  // namespace core
