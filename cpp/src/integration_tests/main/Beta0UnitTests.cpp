#include "beta0/GraphTraits.hpp"
#include "beta0/Manager.hpp"
#include "beta0/ManagerParams.hpp"
#include "core/BasicTypes.hpp"
#include "core/GameServerBase.hpp"
#include "core/SpecTransforms.hpp"
#include "games/connect4/Bindings.hpp"
#include "search/LookupTable.hpp"
#include "search/NNEvalTraits.hpp"
#include "search/NNEvaluation.hpp"
#include "search/SearchLog.hpp"
#include "search/SearchParams.hpp"
#include "search/SearchRequest.hpp"
#include "search/SimpleNNEvaluationService.hpp"
#include "util/BoostUtil.hpp"
#include "util/GTestUtil.hpp"
#include "util/RepoUtil.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using C4Spec = transforms::AddStateStorage<c4::beta0::Spec>;

template <beta0::concepts::Spec Spec>
class MockNNEvaluationService
    : public search::SimpleNNEvaluationService<
        search::NNEvalTraits<beta0::GraphTraits<Spec>, typename Spec::TensorEncodings,
                             search::NNEvaluation<typename Spec::Game, typename Spec::InputFrame,
                                                  typename Spec::NetworkHeads>>> {
 public:
  using Game = Spec::Game;
  using State = Game::State;
  using MoveSet = Game::MoveSet;
  using InputFrame = Spec::InputFrame;
  using NetworkHeads = Spec::NetworkHeads;
  using NNEvaluation = search::NNEvaluation<Game, InputFrame, NetworkHeads>;
  using TensorEncodings = Spec::TensorEncodings;
  using NNEvalTraits =
    search::NNEvalTraits<beta0::GraphTraits<Spec>, TensorEncodings, NNEvaluation>;
  using Base = search::SimpleNNEvaluationService<NNEvalTraits>;
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using GameResultTensor = GameResultEncoding::Tensor;
  using WinShareTensor = TensorEncodings::WinShareTensor;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using NetworkHeadsList = NetworkHeads::List;
  using BackupAccuStaticHead = mp::TypeAt_t<NetworkHeadsList, 5>;
  using BackupAccuStaticTensor = BackupAccuStaticHead::Tensor;
  using Item = Base::Item;

  MockNNEvaluationService() {
    this->set_init_func([&](NNEvaluation* eval, const Item& item) { this->init_eval(eval, item); });
  }

  void init_eval(NNEvaluation* eval, const Item& item) {
    group::element_t sym = group::kIdentity;
    core::seat_index_t seat = item.node()->stable_data().active_seat;

    const State* state_ptr = item.node()->stable_data().get_state();
    RELEASE_ASSERT(state_ptr != nullptr, "state storage must be enabled for beta0 tests");
    MoveSet valid_moves = Game::Rules::analyze(*state_ptr).valid_moves();

    GameResultTensor value;
    value.setZero();  // logits = 0 → neutral evaluation

    PolicyTensor policy;
    policy.setZero();  // logits = 0 → uniform after softmax

    WinShareTensor uncertainty;
    uncertainty.setConstant(0.1f);  // small constant prior uncertainty

    ActionValueTensor action_values;
    action_values.setZero();

    ActionValueTensor action_values_uncertainty;
    action_values_uncertainty.setConstant(0.1f);

    BackupAccuStaticTensor backup_accu_static;
    backup_accu_static.setZero();  // static accumulator portion = 0 (GPU contribution)

    auto outputs = std::make_tuple(policy, value, uncertainty, action_values,
                                   action_values_uncertainty, backup_accu_static);
    using InitParams = NNEvaluation::InitParams;
    InitParams init_params{outputs, valid_moves, item.frame(), sym, seat};
    eval->init(init_params);
  }
};

template <beta0::concepts::Spec Spec>
class ManagerTest : public testing::Test {
 protected:
  using Game = Spec::Game;
  using Manager = beta0::Manager<Spec>;
  using ManagerParams = beta0::ManagerParams<Spec>;
  using Node = beta0::Node<Spec>;
  using Edge = beta0::Edge<Spec>;
  using Move = Game::Move;
  using LookupTable = search::LookupTable<beta0::GraphTraits<Spec>>;
  using InputFrame = Spec::InputFrame;
  using NetworkHeads = Spec::NetworkHeads;
  using NNEvaluation = search::NNEvaluation<Game, InputFrame, NetworkHeads>;
  using NNEvalTraits =
    search::NNEvalTraits<beta0::GraphTraits<Spec>, typename Spec::TensorEncodings, NNEvaluation>;
  using Service = search::NNEvaluationServiceBase<NNEvalTraits>;
  using Service_sptr = Service::sptr;
  using SearchResults = beta0::SearchResults<Spec>;
  using SearchLog = search::SearchLog<beta0::GraphTraits<Spec>>;
  using MockService = MockNNEvaluationService<Spec>;

  static_assert(core::kStoreStates<Spec>, "state-storage required for search-log tests");

 public:
  ManagerTest() : manager_params_(create_manager_params()) {}

  ~ManagerTest() override {
    delete manager_;
    delete search_log_;
  }

  static ManagerParams create_manager_params() {
    ManagerParams params(search::kCompetition);
    params.no_model = true;
    return params;
  }

  void SetUp() override { util::Random::set_seed(0); }

  void init_manager(Service_sptr service = nullptr) {
    core::GameServerBase* server = nullptr;
    manager_ = new Manager(manager_params_, server, service);
    search_log_ = new SearchLog(manager_->lookup_table());
    manager_->set_post_visit_func([&] { search_log_->update(); });
  }

  void start_manager(const std::vector<Move>& initial_moves = {}) {
    manager_->start();
    for (Move move : initial_moves) {
      manager_->update(move);
    }
  }

  ManagerParams& manager_params() { return manager_params_; }

  const SearchResults* search(int num_searches = 0) {
    search::SearchParams search_params(num_searches, true);
    manager_->set_search_params(search_params);
    search::SearchRequest request;
    return manager_->search(request).results;
  }

  SearchLog* get_search_log() { return search_log_; }

  void test_search(const std::string& testname, int num_search,
                   const std::vector<Move>& initial_moves, Service_sptr service,
                   const std::vector<float>& backup_weights = {}) {
    init_manager(service);
    if (!backup_weights.empty()) {
      manager_->set_backup_nn_weights(backup_weights.data(), backup_weights.size());
    }
    start_manager(initial_moves);
    const SearchResults* result = search(num_search);

    auto root = util::Repo::root();
    boost::filesystem::path base_dir = root / "goldenfiles" / "beta0_tests";

    boost::filesystem::path file_path_result = base_dir / (testname + "_result.json");
    boost::filesystem::path file_path_graph = base_dir / (testname + "_graph.json");

    std::stringstream ss_result;
    boost_util::pretty_print(ss_result, result->to_json());

    if (gtest_util::write_goldenfiles) {
      boost_util::write_str_to_file(ss_result.str(), file_path_result);
      boost_util::write_str_to_file(get_search_log()->last_graph_json_str(), file_path_graph);

      boost::filesystem::path log_dir = root / "sample_search_logs" / "beta0_tests";
      boost::filesystem::path log_file_path = log_dir / (testname + "_log.json");
      boost_util::write_str_to_file(get_search_log()->json_str(), log_file_path);
    }

    std::ifstream result_file(file_path_result);
    std::ifstream graph_file(file_path_graph);

    std::string expected_result_json((std::istreambuf_iterator<char>(result_file)),
                                     std::istreambuf_iterator<char>());
    std::string expected_graph_json((std::istreambuf_iterator<char>(graph_file)),
                                    std::istreambuf_iterator<char>());

    EXPECT_EQ(ss_result.str(), expected_result_json);
    EXPECT_EQ(get_search_log()->last_graph_json_str(), expected_graph_json);
  }

 private:
  ManagerParams manager_params_;
  Manager* manager_ = nullptr;
  SearchLog* search_log_ = nullptr;
};

using C4ManagerTest = ManagerTest<C4Spec>;

/*
 * Test 1: BetaZero MCTS with no backup NN loaded — uses the LoTV (Law of Total Variance)
 * path for W estimation.
 */
TEST_F(C4ManagerTest, no_backup_nn) {
  auto service = std::make_shared<MockNNEvaluationService<C4Spec>>();
  test_search("c4_no_backup_nn", 10, {}, service);
}

/*
 * Test 2: BetaZero MCTS with backup NN weights loaded.
 * W_child = 0, W_out = 0, b_out = [0.5, 0.5, 0.1, 0.1].
 * The network override produces Q=[0.5,0.5], W=[0.1,0.1] for all nodes.
 *
 * For c4 beta0: kChildInputDim=5, kBackupHiddenDim=64, kOutputDim=4
 * Weight layout:
 *   W_child [5 * 64 = 320 floats]  → all zero
 *   W_out   [64 * 4 = 256 floats]  → all zero
 *   b_out   [4 floats]             → [0.5, 0.5, 0.1, 0.1]
 * Total: 580 floats
 */
TEST_F(C4ManagerTest, with_backup_nn) {
  using BackupNNEvaluator = beta0::BackupNNEvaluator<C4Spec>;
  constexpr size_t kWeightCount = BackupNNEvaluator::kWeightCount;  // 580 for c4
  constexpr int kOutputDim = BackupNNEvaluator::kOutputDim;         // 4 for c4

  std::vector<float> weights(kWeightCount, 0.0f);
  // Set b_out = [0.5, 0.5, 0.1, 0.1]  (last kOutputDim floats)
  weights[kWeightCount - kOutputDim + 0] = 0.5f;
  weights[kWeightCount - kOutputDim + 1] = 0.5f;
  weights[kWeightCount - kOutputDim + 2] = 0.1f;
  weights[kWeightCount - kOutputDim + 3] = 0.1f;

  auto service = std::make_shared<MockNNEvaluationService<C4Spec>>();
  test_search("c4_with_backup_nn", 10, {}, service, weights);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
