#include <core/concepts/Game.hpp>
#include <core/GameTypes.hpp>
#include <core/BasicTypes.hpp>
#include <games/tictactoe/Game.hpp>
#include <generic_players/MctsPlayer.hpp>
#include <mcts/Manager.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/SearchLog.hpp>

#include <gtest/gtest.h>

template<core::concepts::Game Game>
class MctsPlayerTest : public ::testing::Test {
 protected:

  using Manager = mcts::Manager<Game>;
  using ManagerParams = mcts::ManagerParams<Game>;
  using MctsPlayer = generic::MctsPlayer<Game>;
  using MctsPlayerParams = MctsPlayer::Params;
  using SearchResult = Game::Types::SearchResults;
  using SearchLog = mcts::SearchLog<Game>;
  using PolicyTensor = Game::Types::PolicyTensor;
  using StateHistory = Game::StateHistory;
  using State = Game::State;
  using ActionMask = Game::Types::ActionMask;
    using Service = mcts::NNEvaluationServiceBase<Game>;

 public:
  MctsPlayerTest() : manager_params_(create_manager_params()), player_params_(mcts::kCompetitive) {}

  ManagerParams create_manager_params() {
    ManagerParams params(mcts::kCompetitive);
    params.no_model = true;
    return params;
  }

  void init_manager(Service* service = nullptr) {
    mcts_manager_ = new Manager(manager_params_, service);
    const mcts::SharedData<Game>* shared_data = manager_->shared_data();
    search_log_ = new mcts::SearchLog<Game>(shared_data);
    manager_->set_post_visit_func([&] { search_log_->update(); });
  }

  void start_manager(const std::vector<core::action_t>& initial_actions = {}) {
    manager_->start();
    for (core::action_t action : initial_actions) {
      manager_->shared_data()->update_state(action);
    }
    this->initial_actions_ = initial_actions;
  }

  core::ActionResponse get_action_response(SearchResult* search_result,
                                           PolicyTensor* output_policy) {
    StateHistory state_history =
        mcts_manager_->shared_data()->root_info.history_array[group::kIdentity];
    State state = state_history.current();
    ActionMask valid_actions = Rules::get_legal_moves(state_history);
    core::ActionResponse response =
        mcts_player_->get_action_response(state, valid_actions, search_result, output_policy);
    return response;
  }

  void TearDown() override {
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
TEST_F(TicTacToeMctsPlayerTest, uniform_search_01247) {
  init_manager();
  std::vector<core::action_t> initial_actions = {0, 1, 2, 4, 7};
  start_manager(initial_actions);
  SearchResult search_result;
  PolicyTensor output_policy;
  core::ActionResponse response = get_action_response(&search_result, &output_policy);
}

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}