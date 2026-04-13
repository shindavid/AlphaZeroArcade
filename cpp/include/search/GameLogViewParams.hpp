#pragma once

#include "alpha0/concepts/SpecConcept.hpp"
#include "search/GameLogBase.hpp"
#include "util/FiniteGroups.hpp"

namespace search {

template <::alpha0::concepts::Spec Spec>
struct GameLogViewParams {
  using Game = Spec::Game;
  using InputFrame = Spec::InputFrame;
  using GameLogBase = search::GameLogBase<Spec>;
  using GameLogCompactRecord = GameLogBase::GameLogCompactRecord;
  using GameResultTensor = Spec::TensorEncodings::GameResultEncoding::Tensor;

  const GameLogCompactRecord* record = nullptr;
  const GameLogCompactRecord* next_record = nullptr;
  const InputFrame* cur_frame = nullptr;
  const InputFrame* final_frame = nullptr;
  const GameResultTensor* outcome = nullptr;
  group::element_t sym = group::kIdentity;
};

}  // namespace search
