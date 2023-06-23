#pragma once

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <Eigen/Core>
#include <EigenRand/EigenRand>

#include <core/AbstractSymmetryTransform.hpp>
#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/NeuralNet.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/Constants.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/NNEvaluationService.hpp>
#include <mcts/Node.hpp>
#include <mcts/PUCTStats.hpp>
#include <mcts/SearchParams.hpp>
#include <mcts/SearchResults.hpp>
#include <mcts/SearchThread.hpp>
#include <mcts/SharedData.hpp>
#include <util/AtomicSharedPtr.hpp>
#include <util/BitSet.hpp>
#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>
#include <util/LRUCache.hpp>
#include <util/Math.hpp>
#include <util/Profiler.hpp>

namespace mcts {

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
class Manager {
public:
  using dtype = torch_util::dtype;
  using NNEvaluation = mcts::NNEvaluation<GameState>;
  using NNEvaluation_asptr = typename NNEvaluation::asptr;
  using NNEvaluation_sptr = typename NNEvaluation::sptr;
  using NNEvaluationService = mcts::NNEvaluationService<GameState, Tensorizor>;
  using Node = mcts::Node<GameState, Tensorizor>;
  using PUCTStats = mcts::PUCTStats<GameState, Tensorizor>;
  using SearchThread = mcts::SearchThread<GameState, Tensorizor>;

  using TensorizorTypes = core::TensorizorTypes<Tensorizor>;
  using GameStateTypes = core::GameStateTypes<GameState>;

  static constexpr int kNumPlayers = GameState::kNumPlayers;
  static constexpr int kNumGlobalActions = GameStateTypes::kNumGlobalActions;
  static constexpr int kMaxNumLocalActions = GameState::kMaxNumLocalActions;

  using SearchResults = mcts::SearchResults<GameState>;
  using SymmetryTransform = core::AbstractSymmetryTransform<GameState, Tensorizor>;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using ActionMask = typename GameStateTypes::ActionMask;

  using PolicyArray = typename GameStateTypes::PolicyArray;
  using ValueArray = typename GameStateTypes::ValueArray;
  using LocalPolicyArray = typename GameStateTypes::LocalPolicyArray;

  using InputTensor = typename TensorizorTypes::InputTensor;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using ValueTensor = typename GameStateTypes::ValueTensor;

  using InputShape = typename TensorizorTypes::InputShape;
  using PolicyShape = typename GameStateTypes::PolicyShape;
  using ValueShape = typename GameStateTypes::ValueShape;

  using InputScalar = torch_util::convert_type_t<typename InputTensor::Scalar>;
  using PolicyScalar = torch_util::convert_type_t<typename PolicyTensor::Scalar>;
  using ValueScalar = torch_util::convert_type_t<typename ValueTensor::Scalar>;

  using InputFloatTensor = Eigen::TensorFixedSize<dtype, InputShape, Eigen::RowMajor>;
  using DynamicInputFloatTensor = Eigen::Tensor<dtype, InputShape::count + 1, Eigen::RowMajor>;

private:
  using search_thread_vec_t = std::vector<SearchThread*>;

  class NodeReleaseService {
  public:
    struct work_unit_t {
      work_unit_t(Node* n, Node* a) : node(n), arg(a) {}

      Node* node;
      Node* arg;
    };

    static void release(Node* node, Node* arg=nullptr) { instance_.release_helper(node, arg); }

  private:
    NodeReleaseService();
    ~NodeReleaseService();

    void loop();
    void release_helper(Node* node, Node* arg);

    static NodeReleaseService instance_;

    using work_queue_t = std::vector<work_unit_t>;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    work_queue_t work_queue_[2];
    int queue_index_ = 0;
    int release_count_ = 0;
    int max_queue_size_ = 0;
    bool destructing_ = false;
  };

public:
  /*
   * In multi-threaded mode, the search threads can continue running outside of the main search() method. For example,
   * when playing against a human player, we can continue growing the MCTS tree while the human player thinks.
   */
  static constexpr int kDefaultMaxTreeSize =  4096;

  static int next_instance_id_;  // for naming debug/profiling output files

  Manager(const ManagerParams& params);
  ~Manager();

  int instance_id() const { return shared_data_.manager_id; }
  const ManagerParams& params() const { return params_; }
  int num_search_threads() const { return params().num_search_threads; }
  bool search_active() const { return shared_data_.search_active; }
  NNEvaluationService* nn_eval_service() const { return nn_eval_service_; }

  void start();
  void clear();
  void receive_state_change(core::seat_index_t, const GameState&, core::action_index_t);
  const SearchResults* search(const Tensorizor& tensorizor, const GameState& game_state, const SearchParams& params);

  void start_search_threads(const SearchParams* search_params);
  void wait_for_search_threads();
  void stop_search_threads();
  void run_search(SearchThread* thread, int tree_size_limit);
  void get_cache_stats(int& hits, int& misses, int& size, float& hash_balance_factor) const;

  static float pct_virtual_loss_influenced_puct_calcs() { return NNEvaluationService::pct_virtual_loss_influenced_puct_calcs(); }
  static void end_session() { NNEvaluationService::end_session(); }

private:
  void prune_counts(const SearchParams&);
  static void init_profiling_dir(const std::string& profiling_dir);

  const ManagerParams params_;
  SharedData shared_data_;
  const SearchParams pondering_search_params_;
  search_thread_vec_t search_threads_;
  NNEvaluationService* nn_eval_service_ = nullptr;

  Node* root_ = nullptr;
  SearchResults results_;

  std::mutex search_mutex_;
  std::condition_variable cv_search_;
  int num_active_search_threads_ = 0;
  bool connected_ = false;
};

}  // namespace mcts

#include <core/inl/Mcts.inl>
