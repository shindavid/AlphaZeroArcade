#include "alpha0/ManagerParams.hpp"
#include "alpha0/Traits.hpp"
#include "core/ActionRequest.hpp"
#include "core/BasicTypes.hpp"
#include "core/EvalSpecTransforms.hpp"
#include "core/GameServerBase.hpp"
#include "core/GameStateTree.hpp"
#include "core/StateChangeUpdate.hpp"
#include "core/StateIterator.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "games/tictactoe/Game.hpp"
#include "generic_players/alpha0/Player.hpp"
#include "search/Manager.hpp"
#include "search/SearchLog.hpp"
#include "search/SearchRequest.hpp"
#include "search/TraitsTypes.hpp"
#include "util/BoostUtil.hpp"
#include "util/EigenUtil.hpp"
#include "util/GTestUtil.hpp"
#include "util/RepoUtil.hpp"

#include <gtest/gtest.h>

#include <fstream>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using TicTacToeSpec =
  transforms::AddStateStorage<core::EvalSpec<tictactoe::Game, core::kParadigmAlphaZero>>;

namespace generic::alpha0 {

template <core::concepts::EvalSpec EvalSpec>
class PlayerTest : public ::testing::Test {
 protected:
  using Game = EvalSpec::Game;
  using Traits = ::alpha0::Traits<Game, EvalSpec>;
  using TraitsTypes = search::TraitsTypes<Traits>;
  using InputTensorizor = core::InputTensorizor<Game>;
  using Manager = search::Manager<Traits>;
  using ManagerParams = ::alpha0::ManagerParams<EvalSpec>;
  using Player = generic::alpha0::Player<Traits>;
  using PlayerSharedData = Player::SharedData;
  using PlayerParams = Player::Params;
  using SearchResults = Traits::SearchResults;
  using SearchLog = ::search::SearchLog<Traits>;
  using PolicyTensor = Game::Types::PolicyTensor;
  using State = Game::State;
  using ActionRequest = core::ActionRequest<Game>;
  using StateChangeUpdate = core::StateChangeUpdate<Game>;
  using ActionMask = Game::Types::ActionMask;
  using Service = search::NNEvaluationServiceBase<Traits>;
  using Service_sptr = Service::sptr;
  using Rules = Game::Rules;
  using StateTree = core::GameStateTree<Game>;
  using StateIterator = core::StateIterator<Game>;

 public:
  PlayerTest() : manager_params_(create_manager_params()), player_params_(search::kCompetition) {
    player_params_.num_fast_iters = 10;
    player_params_.num_full_iters = 20;
  }

  ManagerParams create_manager_params() {
    ManagerParams params(search::kCompetition);
    params.no_model = true;
    return params;
  }

  void init(Service_sptr service) {
    core::GameServerBase* server = nullptr;
    auto shared_player_data = std::make_shared<PlayerSharedData>(manager_params_, server, service);
    auto manager = &shared_player_data->manager;
    search_log_ = new SearchLog(manager->lookup_table());
    manager->set_post_visit_func([&] { search_log_->update(); });
    mcts_player_ = new Player(player_params_, shared_player_data, true);
  }

  void start_manager(const std::vector<core::action_t>& initial_actions) {
    mcts_player_->start_game();

    StateTree state_tree;
    state_tree.init();
    core::game_tree_index_t ix = 0;

    for (core::action_t action : initial_actions) {
      core::seat_index_t seat = Rules::get_current_player(state_tree.state(ix));
      ix = state_tree.advance(ix, action);
      StateIterator state_it(&state_tree, ix);
      StateChangeUpdate update(state_it, action, state_tree.get_step(ix), seat);
      mcts_player_->receive_state_change(update);
    }
    initial_actions_ = initial_actions;
  }

  SearchLog* get_search_log() { return search_log_; }

  void test_get_action_policy(const std::string& testname,
                              const std::vector<core::action_t>& initial_actions = {},
                              Service_sptr service = nullptr) {
    init(service);
    start_manager(initial_actions);

    const InputTensorizor& input_tensorizor =
      mcts_player_->get_manager()->root_info()->input_tensorizor;
    const State& state = input_tensorizor.current_state();
    ActionMask valid_actions = Rules::get_legal_moves(state);

    ActionRequest request(state, valid_actions);
    mcts_player_->init_search_mode(request);
    search::SearchRequest search_request;
    const SearchResults* search_results =
      mcts_player_->get_manager()->search(search_request).results;

    PolicyTensor modified_policy = mcts_player_->get_action_policy(search_results, valid_actions);

    std::stringstream ss_result, ss_policy;
    boost_util::pretty_print(ss_result, search_results->to_json());
    boost_util::pretty_print(ss_policy, eigen_util::to_json(modified_policy));

    auto root = util::Repo::root();
    boost::filesystem::path base_dir = root / "goldenfiles" / "generic_players";
    boost::filesystem::path file_path_result = base_dir / (testname + "_result.json");
    boost::filesystem::path file_path_policy = base_dir / (testname + "_policy.json");
    boost::filesystem::path file_path_graph = base_dir / (testname + "_graph.json");

    if (IS_DEFINED(WRITE_GOLDENFILES)) {
      boost_util::write_str_to_file(ss_result.str(), file_path_result);
      boost_util::write_str_to_file(ss_policy.str(), file_path_policy);
      boost_util::write_str_to_file(get_search_log()->last_graph_json_str(), file_path_graph);
    }

    if (IS_DEFINED(WRITE_LOGFILES)) {
      boost::filesystem::path log_dir =
        util::Repo::root() / "sample_search_logs" / "generic_players";
      boost::filesystem::path log_file_path = log_dir / (testname + "_log.json");
      boost_util::write_str_to_file(get_search_log()->json_str(), log_file_path);
    }

    std::ifstream result_file(file_path_result);
    std::ifstream policy_file(file_path_policy);
    std::ifstream graph_file(file_path_graph);

    std::string expected_result_json((std::istreambuf_iterator<char>(result_file)),
                                     std::istreambuf_iterator<char>());
    std::string expected_policy_json((std::istreambuf_iterator<char>(policy_file)),
                                     std::istreambuf_iterator<char>());
    std::string expected_graph_json((std::istreambuf_iterator<char>(graph_file)),
                                    std::istreambuf_iterator<char>());

    EXPECT_EQ(ss_result.str(), expected_result_json);
    EXPECT_EQ(ss_policy.str(), expected_policy_json);
    EXPECT_EQ(get_search_log()->last_graph_json_str(), expected_graph_json);
  }

  void SetUp() override { util::Random::set_seed(1); }

  void TearDown() override {
    delete search_log_;
    delete mcts_player_;
  }

 private:
  ManagerParams manager_params_;
  PlayerParams player_params_;
  Player* mcts_player_;
  SearchLog* search_log_;
  std::vector<core::action_t> initial_actions_;
};

using tictactoe_test = PlayerTest<TicTacToeSpec>;
TEST_F(tictactoe_test, uniform_search) { test_get_action_policy("tictactoe"); }

TEST_F(tictactoe_test, uniform_search_01247) {
  std::vector<core::action_t> initial_actions = {0, 1, 2, 4, 7};
  test_get_action_policy("tictactoe01247", initial_actions);
}

}  // namespace generic::alpha0

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
