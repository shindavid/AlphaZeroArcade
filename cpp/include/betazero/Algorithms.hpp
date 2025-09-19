#pragma once

#include "alphazero/Algorithms.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace beta0 {

// For now, we piggy-back alpha0::Algorithms for most of beta0::Algorithms
template <search::concepts::Traits Traits>
class Algorithms : public alpha0::Algorithms<Traits> {
 public:
  using Base = alpha0::Algorithms<Traits>;
  using EvalSpec = Traits::EvalSpec;
  using Game = Base::Game;
  using GameLogCompactRecord = Base::GameLogCompactRecord;
  using GameLogFullRecord = Base::GameLogFullRecord;
  using GameLogView = Base::GameLogView;
  using GameLogViewParams = Base::GameLogViewParams;
  using GeneralContext = Base::GeneralContext;
  using Edge = Base::Edge;
  using LocalActionValueArray = Base::LocalActionValueArray;
  using LookupTable = Base::LookupTable;
  using Node = Base::Node;
  using SearchContext = Base::SearchContext;
  using SearchResults = Base::SearchResults;
  using TrainingInfoParams = Base::TrainingInfoParams;
  using TrainingInfo = Base::TrainingInfo;
  using TensorData = Base::TensorData;

  static void load_evaluations(SearchContext& context);

  static void write_to_training_info(const TrainingInfoParams&, TrainingInfo& training_info);
  static void to_record(const TrainingInfo&, GameLogFullRecord&);
  static void serialize_record(const GameLogFullRecord& full_record, std::vector<char>& buf);
  static void to_view(const GameLogViewParams&, GameLogView&);

  static void to_results(const GeneralContext&, SearchResults&);

 private:
  static void write_results(const GeneralContext&, const Node* root, group::element_t inv_sym,
                            SearchResults& results);
};

}  // namespace beta0

#include "inline/betazero/Algorithms.inl"
