#pragma once

#include "core/concepts/GameConcept.hpp"
#include "alpha0/VerboseData.hpp"
#include "x0/AuxData.hpp"

namespace alpha0 {

template <core::concepts::Game Game>
struct AuxData : public x0::AuxData {
  using Base = x0::AuxData;
  VerboseData<Game> verbose_data;

  AuxData(const core::ActionResponse& ar, const VerboseData<Game>& vd)
      : Base(ar), verbose_data(vd) {}
};

}  // namespace alpha0
