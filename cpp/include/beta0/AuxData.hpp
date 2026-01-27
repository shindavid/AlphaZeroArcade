#pragma once

#include "core/concepts/GameConcept.hpp"
#include "beta0/VerboseData.hpp"
#include "x0/AuxData.hpp"

#include <memory>

namespace beta0 {

template <core::concepts::Game Game>
struct AuxData : public x0::AuxData {
  using Base = x0::AuxData;
  using Base::Base;
  using VerboseData = beta0::VerboseData<Game>;
  using VerboseData_sptr = std::shared_ptr<VerboseData>;

  VerboseData_sptr verbose_data;
};

}  // namespace alpha0
