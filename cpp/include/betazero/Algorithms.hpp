#pragma once

#include "alphazero/Algorithms.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace beta0 {

// For now, we piggy-back alpha0::Algorithms for most of beta0::Algorithms
template <search::concepts::Traits Traits>
class Algorithms : public alpha0::Algorithms<Traits> {
 public:
  using Base = alpha0::Algorithms<Traits>;

  using Edge = Base::Edge;
  using Game = Base::Game;
  using GameLogCompactRecord = Base::GameLogCompactRecord;
  using GameLogFullRecord = Base::GameLogFullRecord;
  using GameLogView = Base::GameLogView;
  using GameLogViewParams = Base::GameLogViewParams;
  using GeneralContext = Base::GeneralContext;
  using LocalActionValueArray = Base::LocalActionValueArray;
  using LookupTable = Base::LookupTable;
  using Node = Base::Node;
  using NodeStats = Base::NodeStats;
  using SearchContext = Base::SearchContext;
  using SearchResults = Base::SearchResults;
  using TensorData = Base::TensorData;
  using TrainingInfo = Base::TrainingInfo;
  using TrainingInfoParams = Base::TrainingInfoParams;
  using ValueArray = Base::ValueArray;

  static void load_evaluations(SearchContext& context);

  static void to_results(const GeneralContext&, SearchResults&);
  static void write_to_training_info(const TrainingInfoParams&, TrainingInfo& training_info);
  static void to_record(const TrainingInfo&, GameLogFullRecord&);
  static void serialize_record(const GameLogFullRecord& full_record, std::vector<char>& buf);
  static void to_view(const GameLogViewParams&, GameLogView&);

 protected:
  static void init_q(NodeStats& stats, const ValueArray& value, bool pure);
  static void update_q(NodeStats& stats, const ValueArray& value);
};

}  // namespace beta0

#include "inline/betazero/Algorithms.inl"
