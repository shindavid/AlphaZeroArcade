#pragma once

#include "beta0/GameLog.hpp"
#include "core/GameLogBundle.hpp"

namespace core {

template <>
struct GameLogBundle<kParadigmBetaZero> {
  template <typename Spec>
  using GameReadLog = ::beta0::GameReadLog<Spec>;

  template <typename Spec>
  using GameWriteLog = ::beta0::GameWriteLog<Spec>;
};

}  // namespace core
