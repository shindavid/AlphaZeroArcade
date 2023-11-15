#pragma once

#include <bitset>
#include <mutex>
#include <thread>
#include <vector>

#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/Constants.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/NNEvaluationService.hpp>
#include <mcts/Node.hpp>
#include <mcts/NodeCache.hpp>
#include <mcts/PUCTStats.hpp>
#include <mcts/SearchParams.hpp>
#include <mcts/TreeData.hpp>
#include <mcts/TypeDefs.hpp>

namespace mcts {

/*
 * A common base class for PrefetchThread and SearchThread. Members that are used by both classes
 * live here.
 *
 * SearchThread is responsible for updating the "official" counts/values in an MCTS tree. A given
 * tree will only have one active SearchThread at a time.
 *
 * Meanwhile, the tree will have many active PrefetchThread's, which do the "hard" work. Namely,
 * they zip around the tree, expand children, and make batched neural network evaluations. When
 * doing this, they try to predict in advance which parts of the tree the single SearchThread will
 * visit in the future, similarly to how hardware prefetchers try to predict which parts of memory
 * the CPU will access in the future. The way they try to predict this is by mimicking the same
 * PUCT calculations that the SearchThread will do, except they apply virtual loss to make use of
 * parallelism.
 *
 * For the most part, the PrefetchThread's operate independently of the single SearchThread.
 * However, there is a mechanism by which the SearchThread can alert the PrefetchThread's if it
 * detects that the PrefetchThread's are wasting too much time predicting in the wrong region of the
 * tree. In such scenarios, the PrefetchThread's will stop and recalibrate according to the
 * SearchThread's "official" counts/values.
 *
 * In typical multithreaded MCTS implementations, these roles are effectively merged into one -
 * each thread does "prefetching"-style behavior, applying virtual loss to make use of parallelism,
 * but then uses the results of the calculations that supports its prefetch-predictions to directly
 * update the "official" counts/values.
 *
 * The following text, taken from a message from user nerai from the Computer Go Community Discord,
 * summarizes the problem with such implementations:
 *
 * "There is an inherent problem with parallelizing MCTS. The algorithm was proven to converge to
 * optimal results in serial operation, but in parallel operation, this convergence is degraded.
 * This is perhaps mostly due to the fact that the algorithm expects a node to be visited only when
 * it is the most interesting at some point in the search. By definition, a 2-parallel search must
 * look at (in the best case) the second most interesting node concurrently to the most interesting
 * node, and so on.
 *
 * The issue with this is that a node with more visits receives a greater weight in further
 * calculations. As a simplified example, imagine a node that has one excellent continuation while
 * all other child moves are horrible. Of course, the good path should be taken almost exclusively.
 * But if this node is evaluated in parallel, in addition to the good path the algorithm must pick
 * some bad paths, too, to avoid contention.
 *
 * The results of these evaluations will be somehow averaged, leading to the parent node looking
 * worse than it should. It will thus take some time for the search to focus on the parent node
 * again and eventually recognize its dominant continuation. Until then, the search is -- to some
 * degree -- misguided and will produce worse results. This can drastically reduce playing strength.
 *
 * All of this does not matter much at a parallelism level of 5 or 10, as it will work around the
 * inaccuracy over a few thousand visits, which is common nowadays. It has, however, a severe impact
 * at levels of hundreds, thousands or more. MCTS in its classic variants is not built to deal with
 * this."
 */
template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
class TreeTraversalThread {
 public:
  using GameStateTypes = core::GameStateTypes<GameState>;
  using NNEvaluation = mcts::NNEvaluation<GameState>;
  using NNEvaluationService = mcts::NNEvaluationService<GameState, Tensorizor>;
  using Node = mcts::Node<GameState, Tensorizor>;
  using NodeCache = mcts::NodeCache<GameState, Tensorizor>;
  using PUCTStats = mcts::PUCTStats<GameState, Tensorizor>;
  using TreeData = mcts::TreeData<GameState, Tensorizor>;
  using edge_t = typename Node::edge_t;

  using Action = typename GameStateTypes::Action;
  using ActionMask = typename GameStateTypes::ActionMask;
  using LocalPolicyArray = typename GameStateTypes::LocalPolicyArray;
  using NNEvaluation_sptr = typename NNEvaluation::sptr;
  using PolicyShape = typename GameStateTypes::PolicyShape;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using ValueArray = typename GameStateTypes::ValueArray;
  using ValueTensor = typename GameStateTypes::ValueTensor;

  static constexpr int kNumPlayers = GameState::kNumPlayers;
  static constexpr int kNumGlobalActionsBound = GameStateTypes::kNumGlobalActionsBound;

  using dtype = torch_util::dtype;
  using profiler_t = search_thread_profiler_t;

  TreeTraversalThread(TreeTraversalMode traversal_mode,
                      const boost::filesystem::path& profiling_dir, int thread_id);
  ~TreeTraversalThread();

  int thread_id() const { return thread_id_; }

  void join();
  void kill();
  bool is_pondering() const { return search_params_->ponder; }

  void dump_profiling_stats() { profiler_.dump(64); }

 protected:
  struct VirtualIncrement {
    void operator()(Node* node, TreeTraversalMode mode) const {
      node->stats(mode).virtual_increment();
    }
  };

  struct RealIncrement {
    void operator()(Node* node, TreeTraversalMode mode) const {
      node->stats(mode).real_increment();
    }
  };

  struct IncrementTransfer {
    void operator()(Node* node, TreeTraversalMode mode) const {
      node->stats(mode).increment_transfer();
    }
  };

  struct RealIncrementAndDeduceCertainOutcomes {
    RealIncrementAndDeduceCertainOutcomes(const ValueArray& value) : value(value) {}
    void operator()(Node* node, TreeTraversalMode mode) const {
      node->stats(mode).real_increment();
      node->stats(mode).deduce_certain_outcomes(value);
    }
    const ValueArray& value;
  };

  struct evaluation_result_t {
    NNEvaluation_sptr evaluation;
    bool backpropagated_virtual_loss;
  };

  struct visitation_t {
    visitation_t(Node* n, edge_t* e) : node(n), edge(e) {}
    Node* node;
    edge_t* edge;
  };
  using search_path_t = std::vector<visitation_t>;

  void add_dirichlet_noise(LocalPolicyArray& P);
  void backprop(const ValueArray& value, BackpropMode mode);
  void short_circuit_backprop(edge_t* last_edge);
  std::string search_path_str() const;  // slow, for debugging

  /*
   * Used in visit().
   *
   * Applies PUCT criterion to select the best child-index to visit from the given Node.
   *
   * TODO: as we experiment with things like auxiliary NN output heads, dynamic cPUCT values,
   * etc., this method will evolve. It probably makes sense to have the behavior as part of the
   * Tensorizor, since there is coupling with NN architecture (in the form of output heads).
   */
  core::action_index_t get_best_action_index(Node* node, NNEvaluation* evaluation);

  auto& dirichlet_gen() { return tree_data_->dirichlet_gen(); }
  auto& rng() { return tree_data_->rng(); }
  float root_softmax_temperature() const { return tree_data_->root_softmax_temperature_value(); }

  TreeData* tree_data_ = nullptr;
  NNEvaluationService* nn_eval_service_ = nullptr;
  const SearchParams* search_params_ = nullptr;
  const ManagerParams* manager_params_ = nullptr;

  search_path_t search_path_;
  profiler_t profiler_;

  const TreeTraversalMode traversal_mode_;
  const int thread_id_;
  std::thread* thread_ = nullptr;
};

}  // namespace mcts

#include <mcts/inl/TreeTraversalThread.inl>
