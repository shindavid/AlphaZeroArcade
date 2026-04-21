#pragma once

#include "alpha0/GameLog.hpp"
#include "core/GameLogBundle.hpp"

namespace core {

// TODO: change these to use ::beta0::* types

template <>
struct GameLogBundle<kParadigmBetaZero> {
  template <typename Spec>
  using GameReadLog = ::alpha0::GameReadLog<Spec>;

  template <typename Spec>
  using GameWriteLog = ::alpha0::GameWriteLog<Spec>;
};

}  // namespace core
