#include <core/concepts/Game.hpp>
#include <core/GameTypes.hpp>
#include <core/BasicTypes.hpp>
#include <games/tictactoe/Game.hpp>
#include <generic_players/MctsPlayer.hpp>
#include <mcts/Manager.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/SearchLog.hpp>
#include <util/EigenUtil.hpp>
#include <util/BoostUtil.hpp>

#include <gtest/gtest.h>

#include <fstream>

template<core::concepts::Game Game>
class MctsPlayerTest : public ::testing::Test {
 protected:

  using Manager = mcts::Manager<Game>;
  using ManagerParams = mcts::ManagerParams<Game>;
  using MctsPlayer = generic::MctsPlayer<Game>;
  using MctsPlayerParams = MctsPlayer::Params;
  using SearchResults = Game::Types::SearchResults;
  using SearchLog = mcts::SearchLog<Game>;
  using PolicyTensor = Game::Types::PolicyTensor;
  using StateHistory = Game::StateHistory;
  using State = Game::State;
  using ActionMask = Game::Types::ActionMask;
  using Service = mcts::NNEvaluationServiceBase<Game>;
  using Rules = Game::Rules;

 public:
  MctsPlayerTest() : manager_params_(create_manager_params()), player_params_(mcts::kCompetitive) {}

  ManagerParams create_manager_params() {
    ManagerParams params(mcts::kCompetitive);
    params.no_model = true;
    return params;
  }

  void init(Service* service = nullptr) {
    mcts_manager_ = new Manager(manager_params_, service);
    const mcts::SharedData<Game>* shared_data = mcts_manager_->shared_data();
    search_log_ = new mcts::SearchLog<Game>(shared_data);
    mcts_manager_->set_post_visit_func([&] { search_log_->update(); });
    mcts_manager_->start();
    mcts_manager_->set_player_data(mcts_manager_->shared_data());
    mcts_player_ = new MctsPlayer(player_params_, mcts_manager_);
  }

  void start_manager(const std::vector<core::action_t>& initial_actions) {
    mcts_player_->start_game();
    mcts_manager_->start_threads();
    for (core::action_t action : initial_actions) {
      auto shared_data = mcts_manager_->shared_data();
      shared_data->update_state(action);
    }
    initial_actions_ = initial_actions;
  }

  mcts::SearchLog<Game>* get_search_log() { return search_log_; }

  void test_get_action_policy(std::string testname,
                              const std::vector<core::action_t>& initial_actions = {}) {
    init();
    start_manager(initial_actions);

    StateHistory state_history =
        mcts_manager_->shared_data()->root_info.history_array[group::kIdentity];
    ActionMask valid_actions = Rules::get_legal_moves(state_history);

    core::SearchMode search_mode = mcts_player_->choose_search_mode();
    const SearchResults* search_result = mcts_player_->mcts_search(search_mode);

    PolicyTensor modified_policy =
        mcts_player_->get_action_policy(search_mode, search_result, valid_actions);

    std::stringstream ss_result, ss_policy;
    boost_util::pretty_print(ss_result, search_result->to_json());
    boost_util::pretty_print(ss_policy, eigen_util::to_json(modified_policy));

    boost::filesystem::path file_path_result =
        util::Repo::root() / "goldenfiles" / "generic_players" / (testname + "_result.json");

    boost::filesystem::path file_path_policy =
        util::Repo::root() / "goldenfiles" / "generic_players" / (testname + "_policy.json");

    boost::filesystem::path file_path_log =
        util::Repo::root() / "goldenfiles" / "generic_players" / (testname + "_log.json");

    if (IS_MACRO_ENABLED(WRITE_GOLDENFILES)) {
      boost_util::write_str_to_file(ss_result.str(), file_path_result);
      boost_util::write_str_to_file(ss_policy.str(), file_path_policy);
      boost_util::write_str_to_file(get_search_log()->json_str(), file_path_log);
    }

    std::ifstream result_file(file_path_result);
    std::ifstream policy_file(file_path_policy);
    std::ifstream log_file(file_path_log);

    std::string expected_result_json((std::istreambuf_iterator<char>(result_file)),
                                     std::istreambuf_iterator<char>());
    std::string expected_policy_json((std::istreambuf_iterator<char>(policy_file)),
                                     std::istreambuf_iterator<char>());
    std::string expected_log_json((std::istreambuf_iterator<char>(log_file)),
                                  std::istreambuf_iterator<char>());

    EXPECT_EQ(ss_result.str(), expected_result_json);
    EXPECT_EQ(ss_policy.str(), expected_policy_json);
    EXPECT_EQ(get_search_log()->json_str(), expected_log_json);
  }

  void TearDown() override {
    delete search_log_;
    delete mcts_manager_;
    delete mcts_player_;
  }

private:
  ManagerParams manager_params_;
  MctsPlayerParams player_params_;
  Manager* mcts_manager_;
  MctsPlayer* mcts_player_;
  mcts::SearchLog<Game>* search_log_;
  std::vector<core::action_t> initial_actions_;
};

using TicTacToeMctsPlayerTest = MctsPlayerTest<tictactoe::Game>;
TEST_F(TicTacToeMctsPlayerTest, uniform_search) {
  test_get_action_policy("tictactoe");
}

using TicTacToeMctsPlayerTest = MctsPlayerTest<tictactoe::Game>;
TEST_F(TicTacToeMctsPlayerTest, uniform_search_01247) {
  std::vector<core::action_t> initial_actions = {0, 1, 2, 4, 7};
  test_get_action_policy("tictactoe01247", initial_actions);
}

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}