#pragma once

#include "alpha0/GameLog.hpp"
#include "core/GameLogBundle.hpp"

namespace core {

template <>
struct GameLogBundle<kParadigmAlphaZero> {
  template <typename Spec>
  using GameReadLog = ::alpha0::GameReadLog<Spec>;

  template <typename Spec>
  using GameWriteLog = ::alpha0::GameWriteLog<Spec>;
};

}  // namespace core
