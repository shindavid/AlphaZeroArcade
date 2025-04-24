#include <core/concepts/Game.hpp>
#include <core/BasicTypes.hpp>
#include <core/GameTypes.hpp>
#include <core/tests/Common.hpp>
#include <games/GameTransforms.hpp>
#include <games/tictactoe/Game.hpp>
#include <generic_players/MctsPlayer.hpp>
#include <mcts/Manager.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/SearchLog.hpp>
#include <util/BoostUtil.hpp>
#include <util/EigenUtil.hpp>
#include <util/GTestUtil.hpp>
#include <util/LoggingUtil.hpp>

#include <gtest/gtest.h>

#include <fstream>

namespace generic {

using TicTacToe = game_transform::AddStateStorage<tictactoe::Game>;

template <core::concepts::Game Game>
class MctsPlayerTest : public ::testing::Test {
 protected:
  using Manager = mcts::Manager<Game>;
  using ManagerParams = mcts::ManagerParams<Game>;
  using MctsPlayer = generic::MctsPlayer<Game>;
  using MctsPlayerSharedData = MctsPlayer::SharedData;
  using MctsPlayerParams = MctsPlayer::Params;
  using SearchResults = Game::Types::SearchResults;
  using SearchLog = mcts::SearchLog<Game>;
  using PolicyTensor = Game::Types::PolicyTensor;
  using StateHistory = Game::StateHistory;
  using State = Game::State;
  using ActionRequest = Game::Types::ActionRequest;
  using ActionResponse = Game::Types::ActionResponse;
  using ActionMask = Game::Types::ActionMask;
  using Service = mcts::NNEvaluationServiceBase<Game>;
  using Rules = Game::Rules;

 public:
  MctsPlayerTest() : manager_params_(create_manager_params()), player_params_(mcts::kCompetitive) {
    player_params_.num_fast_iters = 10;
    player_params_.num_full_iters = 20;
  }

  ManagerParams create_manager_params() {
    ManagerParams params(mcts::kCompetitive);
    params.no_model = true;
    return params;
  }

  void init(Service* service) {
    auto shared_player_data = std::make_shared<MctsPlayerSharedData>(manager_params_, service);
    auto manager = &shared_player_data->manager;
    search_log_ = new mcts::SearchLog<Game>(manager->lookup_table());
    manager->set_post_visit_func([&] { search_log_->update(); });
    mcts_player_ = new MctsPlayer(player_params_, shared_player_data, true);
  }

  void start_manager(const std::vector<core::action_t>& initial_actions) {
    mcts_player_->start_game();
    StateHistory history;
    history.initialize(Rules{});
    for (core::action_t action : initial_actions) {
      core::seat_index_t seat = Rules::get_current_player(history.current());
      Rules::apply(history, action);
      mcts_player_->receive_state_change(seat, history.current(), action);
    }
    initial_actions_ = initial_actions;
  }

  mcts::SearchLog<Game>* get_search_log() { return search_log_; }

  void test_get_action_policy(const std::string& testname,
                              const std::vector<core::action_t>& initial_actions = {},
                              Service* service = nullptr) {
    init(service);
    start_manager(initial_actions);

    const StateHistory& state_history =
        mcts_player_->get_manager()->root_info()->history_array[group::kIdentity];
    ActionMask valid_actions = Rules::get_legal_moves(state_history);

    ActionRequest request(state_history.current(), valid_actions);
    mcts_player_->init_search_mode(request);
    const SearchResults* search_results = mcts_player_->get_manager()->search();

    PolicyTensor modified_policy =
        mcts_player_->get_action_policy(search_results, valid_actions);

    std::stringstream ss_result, ss_policy;
    boost_util::pretty_print(ss_result, search_results->to_json());
    boost_util::pretty_print(ss_policy, eigen_util::to_json(modified_policy));

    boost::filesystem::path base_dir = util::Repo::root() / "goldenfiles" / "generic_players";
    boost::filesystem::path file_path_result = base_dir / (testname + "_result.json");
    boost::filesystem::path file_path_policy = base_dir / (testname + "_policy.json");
    boost::filesystem::path file_path_graph = base_dir / (testname + "_graph.json");

    if (IS_MACRO_ENABLED(WRITE_GOLDENFILES)) {
      boost_util::write_str_to_file(ss_result.str(), file_path_result);
      boost_util::write_str_to_file(ss_policy.str(), file_path_policy);
      boost_util::write_str_to_file(get_search_log()->last_graph_json_str(), file_path_graph);
    }

    if (IS_MACRO_ENABLED(WRITE_LOGFILES)) {
      boost::filesystem::path log_dir = util::Repo::root() / "sample_search_logs" / "generic_players";
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
  MctsPlayerParams player_params_;
  MctsPlayer* mcts_player_;
  mcts::SearchLog<Game>* search_log_;
  std::vector<core::action_t> initial_actions_;
};

using tictactoe_test = MctsPlayerTest<TicTacToe>;
TEST_F(tictactoe_test, uniform_search) {
  test_get_action_policy("tictactoe");
}

TEST_F(tictactoe_test, uniform_search_01247) {
  std::vector<core::action_t> initial_actions = {0, 1, 2, 4, 7};
  test_get_action_policy("tictactoe01247", initial_actions);
}

}  // namespace generic

int main(int argc, char** argv) {
  return launch_gtest(argc, argv);
}
