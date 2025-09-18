#pragma once

#include "search/AlgorithmsBase.hpp"

namespace beta0 {

// For now, most of the code lives in AlgorithmsBase, because beta0 is currently just a copy of
// alpha0. As we specialize beta0 more, we should move more code from AlgorithmsBase to
// alpha0::Algorithms.
template <search::concepts::Traits Traits>
class Algorithms : public search::AlgorithmsBase<Traits> {
 public:
  using Base = search::AlgorithmsBase<Traits>;
  using Game = Base::Game;
  using GameLogCompactRecord = Base::GameLogCompactRecord;
  using GameLogFullRecord = Base::GameLogFullRecord;
  using GameLogView = Base::GameLogView;
  using GameLogViewParams = Base::GameLogViewParams;
  using SearchResults = Base::SearchResults;
  using TrainingInfoParams = Base::TrainingInfoParams;
  using TrainingInfo = Base::TrainingInfo;
  using TensorData = Base::TensorData;

  static void write_to_training_info(const TrainingInfoParams&, TrainingInfo& training_info);
  static void to_record(const TrainingInfo&, GameLogFullRecord&);
  static void serialize_record(const GameLogFullRecord& full_record, std::vector<char>& buf);
  static void to_view(const GameLogViewParams&, GameLogView&);
};

}  // namespace beta0

#include "inline/betazero/Algorithms.inl"
