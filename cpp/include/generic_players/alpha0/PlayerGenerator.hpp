#pragma once

// Redirect: generic::alpha0::{PlayerGeneratorBase, CompetitionPlayerGenerator, ...} → alpha0::

#include "alpha0/PlayerGenerator.hpp"

namespace generic::alpha0 {

template <typename PlayerT, search::Mode Mode>
using PlayerGeneratorBase = ::alpha0::PlayerGeneratorBase<PlayerT, Mode>;

template <typename PlayerT>
using CompetitionPlayerGenerator = ::alpha0::CompetitionPlayerGenerator<PlayerT>;

template <typename PlayerT>
using TrainingPlayerGenerator = ::alpha0::TrainingPlayerGenerator<PlayerT>;

template <typename GeneratorT>
using Subfactory = ::alpha0::Subfactory<GeneratorT>;

}  // namespace generic::alpha0
