#pragma once

#include "search/GameLogBase.hpp"
#include "search/concepts/SearchSpecConcept.hpp"
#include "util/FiniteGroups.hpp"

namespace search {

template <search::concepts::SearchSpec SearchSpec>
struct GameLogViewParams {
  using Game = SearchSpec::Game;
  using EvalSpec = SearchSpec::EvalSpec;
  using InputFrame = EvalSpec::InputFrame;
  using GameLogBase = search::GameLogBase<SearchSpec>;
  using GameLogCompactRecord = GameLogBase::GameLogCompactRecord;
  using GameResultTensor = EvalSpec::TensorEncodings::GameResultEncoding::Tensor;

  const GameLogCompactRecord* record = nullptr;
  const GameLogCompactRecord* next_record = nullptr;
  const InputFrame* cur_frame = nullptr;
  const InputFrame* final_frame = nullptr;
  const GameResultTensor* outcome = nullptr;
  group::element_t sym = group::kIdentity;
};

}  // namespace search
