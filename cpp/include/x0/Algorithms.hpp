#pragma once

#include "core/ActionResponse.hpp"
#include "core/ActionSymmetryTable.hpp"
#include "search/GeneralContext.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchContext.hpp"
#include "search/TrainingDataWriter.hpp"
#include "search/concepts/SearchSpecConcept.hpp"

namespace x0 {

// Base class of {alpha0,beta0}::Algorithms
template <search::concepts::SearchSpec SearchSpec>
class Algorithms {
 public:
  using Game = SearchSpec::Game;
  using Edge = SearchSpec::Edge;
  using TrainingInfo = SearchSpec::TrainingInfo;
  using SearchResults = SearchSpec::SearchResults;
  using SearchContext = search::SearchContext<SearchSpec>;
  using GeneralContext = search::GeneralContext<SearchSpec>;
  using LookupTable = search::LookupTable<SearchSpec>;

  using State = Game::State;
  using Node = SearchSpec::Node;

  using ActionResponse = core::ActionResponse<Game>;
  using EvalSpec = SearchSpec::EvalSpec;
  using TensorEncodings = EvalSpec::TensorEncodings;
  using PolicyEncoding = TensorEncodings::PolicyEncoding;
  using ActionValueEncoding = TensorEncodings::ActionValueEncoding;
  using PolicyTensor = PolicyEncoding::Tensor;
  using ActionValueTensor = ActionValueEncoding::Tensor;
  using ActionSymmetryTable = core::ActionSymmetryTable<EvalSpec>;
  using Symmetries = EvalSpec::Symmetries;
  using InputFrame = EvalSpec::InputFrame;

  using TrainingDataWriter = search::TrainingDataWriter<SearchSpec>;
  using GameWriteLog = TrainingDataWriter::GameWriteLog;
  using GameWriteLog_sptr = TrainingDataWriter::GameWriteLog_sptr;

  static void print_visit_info(const SearchContext&);
  static void write_to_training_info(bool use_for_training, const ActionResponse& response,
                                     const SearchResults*, core::seat_index_t seat,
                                     GameWriteLog_sptr, TrainingInfo& training_info);

 protected:
  static bool validate_and_symmetrize_policy_target(const SearchResults* mcts_results,
                                                    PolicyTensor& target);
  static void load_action_symmetries(const GeneralContext&, const Node* root, SearchResults&);
  static ActionValueTensor apply_mask(const ActionValueTensor&, const PolicyTensor& mask,
                                      float invalid_value = -1.0f);
};

}  // namespace x0

#include "inline/x0/Algorithms.inl"
