/*
 * In the past, we used this to generate perfectly-labeled training data for testing purposes. Conceivably, this
 * could still be useful, but it's no longer used now that the AlphaZero complete loop works.
 *
 * After some refactoring (the introduction of GameServer to replace GameRunner/ParallelGameRunner), this no longer
 * compiles. Because it's no longer used, I'm not going to bother fixing it. But I will leave it around in case we want
 * to revive it for whatever reason.
 */
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <core/AbstractPlayer.hpp>
#include <core/DerivedTypes.hpp>
#include <core/ParallelGameRunner.hpp>
#include <core/TrainingDataWriter.hpp>
#include <common/players/RandomPlayer.hpp>
#include <games/connect4/Constants.hpp>
#include <games/connect4/GameState.hpp>
#include <games/connect4/players/PerfectPlayer.hpp>
#include <games/connect4/Tensorizor.hpp>

namespace bf = boost::filesystem;

using seat_index_t = core::seat_index_t;

using GameState = c4::GameState;
using Tensorizor = c4::Tensorizor;
using Player = core::AbstractPlayer<GameState>;
using TrainingDataWriter = core::TrainingDataWriter<GameState, Tensorizor>;
using ParallelGameRunner = core::ParallelGameRunner<GameState>;
using player_array_t = Player::player_array_t;

using GameStateTypes = core::GameStateTypes<GameState>;
using GameOutcome = GameStateTypes::GameOutcome;
using Action = GameStateTypes::Action;
using ActionMask = GameStateTypes::ActionMask;

using RandomPlayer = core::RandomPlayer<GameState>;

class OracleSupervisor {
public:
  OracleSupervisor(TrainingDataWriter* writer, const c4::PerfectPlayParams& perfect_play_params)
  : oracle_(perfect_play_params)
  , writer_(writer) {}

  TrainingDataWriter::GameData_sptr start_game(core::game_id_t game_id) {
    tensorizor_.clear();
    move_history_.reset();
    return writer_->get_data(game_id);
  }

  void write(TrainingDataWriter::GameData_sptr game_data, const GameState& state) {
    auto query_result = oracle_.query(move_history_);
    int best_score = query_result.score;
    const ActionMask& moves = best_score > 0 ? query_result.good_moves : query_result.best_moves;

    float cur_player_value = best_score > 0 ? +1 : (best_score < 0 ? 0 : 0.5f);

    auto slab = game_data->get_next_slab();
    auto& input = slab.input;
    auto& value = slab.value;
    auto& policy = slab.policy;

    seat_index_t cp = state.get_current_player();
    value(cp) = cur_player_value;
    value(1 - cp) = 1 - cur_player_value;
    for (size_t k = 0; k < moves.size(); ++k) {
      policy.data()[k] = moves[k];
    }

    tensorizor_.tensorize(input, state);
  }

  void close(TrainingDataWriter::GameData_sptr game_data) {
    writer_->close(game_data);
  }

  void receive_move(const GameState& state, const Action& action) {
    tensorizor_.receive_state_change(state, action);
    move_history_.append(action[0]);
  }

private:
  c4::PerfectOracle oracle_;
  TrainingDataWriter* writer_;
  Tensorizor tensorizor_;
  c4::PerfectOracle::MoveHistory move_history_;
};

template<class BasePlayer>
class OracleSupervisedPlayer : public BasePlayer {
public:
  OracleSupervisedPlayer(TrainingDataWriter* writer, const c4::PerfectPlayParams& perfect_play_params)
  : supervisor_(new OracleSupervisor(writer, perfect_play_params))
  , owns_supervisor_(true) {}

  OracleSupervisedPlayer(OracleSupervisor* supervisor)
  : supervisor_(supervisor)
  , owns_supervisor_(false) {}

  ~OracleSupervisedPlayer() {
    if (owns_supervisor_) delete supervisor_;
  }

  OracleSupervisor* supervisor() const { return supervisor_; }

  void start_game() override {
    BasePlayer::start_game();
    game_data_ = supervisor_->start_game(BasePlayer::get_game_id());
  }

  void receive_state_change(
      seat_index_t p, const GameState& state, const Action& action,
      const GameOutcome& outcome) override
  {
    BasePlayer::receive_state_change(p, state, action, outcome);
    if (core::is_terminal_outcome(outcome)) {
      supervisor_->close(game_data_);
    } else if (p == BasePlayer::get_my_seat()) {
      supervisor_->receive_move(state, action);
    }
  }

  Action get_action(const GameState& state, const ActionMask& mask) override {
    supervisor_->write(game_data_, state);
    return BasePlayer::get_action(state, mask);
  }

private:
  OracleSupervisor* supervisor_;
  TrainingDataWriter::GameData_sptr game_data_;
  bool owns_supervisor_;
};

player_array_t create_players(TrainingDataWriter* writer, const c4::PerfectPlayParams& perfect_play_params) {
  using player_t = OracleSupervisedPlayer<RandomPlayer>;
  player_t* p1 = new player_t(writer, perfect_play_params);
  player_t* p2 = new player_t(p1->supervisor());
  return player_array_t{p1, p2};;
}

ParallelGameRunner::Params get_default_parallel_game_runner_params() {
  ParallelGameRunner::Params parallel_game_runner_params;
  parallel_game_runner_params.num_games = 10000;
  parallel_game_runner_params.parallelism = 24;
  parallel_game_runner_params.display_progress_bar = true;
  return parallel_game_runner_params;
}

int main(int ac, char* av[]) {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  c4::PerfectPlayParams perfect_play_params;
  ParallelGameRunner::register_signal(SIGTERM);
  ParallelGameRunner::Params parallel_game_runner_params = get_default_parallel_game_runner_params();
  TrainingDataWriter::Params training_data_writer_params;

  po2::options_description raw_desc("General options");
  auto desc = raw_desc.template add_option<"help", 'h'>("help")
      .add(perfect_play_params.make_options_description())
      .add(parallel_game_runner_params.make_options_description())
      .add(training_data_writer_params.make_options_description());

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  TrainingDataWriter writer(training_data_writer_params);
  ParallelGameRunner runner(parallel_game_runner_params);
  runner.register_players([&]() { return create_players(&writer, perfect_play_params); });
  runner.run();

  printf("\nWrote data to: %s\n", training_data_writer_params.games_dir.c_str());
  return 0;
}
