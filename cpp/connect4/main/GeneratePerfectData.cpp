#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <common/AbstractPlayer.hpp>
#include <common/DerivedTypes.hpp>
#include <common/ParallelGameRunner.hpp>
#include <common/RandomPlayer.hpp>
#include <common/TrainingDataWriter.hpp>
#include <connect4/C4Constants.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4PerfectPlayer.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <util/BitSet.hpp>

namespace bf = boost::filesystem;

struct Args {
  std::string games_dir_str;
};

using player_index_t = common::player_index_t;
using action_index_t = common::action_index_t;

using GameState = c4::GameState;
using Tensorizor = c4::Tensorizor;
using Player = common::AbstractPlayer<GameState>;
using TrainingDataWriter = common::TrainingDataWriter<GameState, Tensorizor>;
using ParallelGameRunner = common::ParallelGameRunner<GameState>;
using player_array_t = Player::player_array_t;

using GameStateTypes = common::GameStateTypes<GameState>;
using GameOutcome = GameStateTypes::GameOutcome;
using ActionMask = GameStateTypes::ActionMask;

using RandomPlayer = common::RandomPlayer<GameState>;

class OracleSupervisor {
public:
  OracleSupervisor(TrainingDataWriter* writer)
  : writer_(writer) {}

  void start_game() {
    game_data_ = writer_->allocate_data();
    tensorizor_.clear();
    move_history_.reset();
  }

  void write(const GameState& state) {
    auto query_result = oracle_.get_best_moves(move_history_);
    const ActionMask& best_moves = query_result.moves;
    int best_score = query_result.score;

    float cur_player_value = best_score > 0 ? +1 : (best_score < 0 ? 0 : 0.5f);

    auto slab = game_data_->get_next_slab();
    auto& input = slab.input;
    auto& value = slab.value;
    auto& policy = slab.policy;

    player_index_t cp = state.get_current_player();
    value(cp) = cur_player_value;
    value(1 - cp) = 1 - cur_player_value;
    for (size_t k = 0; k < best_moves.size(); ++k) {
      policy.data()[k] = best_moves[k];
    }

    tensorizor_.tensorize(input, state);
  }

  void receive_move(const GameState& state, action_index_t action, const GameOutcome& outcome) {
    if (common::is_terminal_outcome(outcome)) {
      writer_->process(game_data_);
    } else {
      tensorizor_.receive_state_change(state, action);
      move_history_.append(action);
    }
  }

private:
  c4::PerfectOracle oracle_;
  TrainingDataWriter* writer_;
  TrainingDataWriter::GameData* game_data_;
  Tensorizor tensorizor_;
  c4::PerfectOracle::MoveHistory move_history_;
};

template<class BasePlayer>
class OracleSupervisedPlayer : public BasePlayer {
public:
  OracleSupervisedPlayer(OracleSupervisor* supervisor, bool primary)
  : supervisor_(supervisor)
  , primary_(primary) {}

  ~OracleSupervisedPlayer() {
    if (primary_) delete supervisor_;
  }

  void start_game(const player_array_t& players, player_index_t seat_assignment) override {
    BasePlayer::start_game(players, seat_assignment);
    if (!primary_) return;
    supervisor_->start_game();
  }

  void receive_state_change(
      player_index_t p, const GameState& state, action_index_t action,
      const GameOutcome& outcome) override
  {
    BasePlayer::receive_state_change(p, state, action, outcome);
    if (!primary_) return;
    supervisor_->receive_move(state, action, outcome);
  }

  action_index_t get_action(const GameState& state, const ActionMask& mask) override {
    supervisor_->write(state);
    return BasePlayer::get_action(state, mask);
  }

private:
  OracleSupervisor* supervisor_;
  bool primary_;
};

player_array_t create_players(TrainingDataWriter* writer) {
  OracleSupervisor* supervisor = new OracleSupervisor(writer);
  using player_t = OracleSupervisedPlayer<RandomPlayer>;
  Player* p1 = new player_t(supervisor, true);
  Player* p2 = new player_t(supervisor, false);
  return player_array_t{p1, p2};;
}

int main(int ac, char* av[]) {
  Args args;

  namespace po = boost::program_options;
  po::options_description desc("Generate training data labeled by oracle");
  desc.add_options()("help,h", "help");

  c4::PerfectPlayParams::PerfectPlayParams::add_options(desc, true);
  ParallelGameRunner::global_params_.num_games = 10000;
  ParallelGameRunner::global_params_.parallelism_factor = 24;
  ParallelGameRunner::global_params_.display_progress_bar = true;
  ParallelGameRunner::add_options(desc, true);

  desc.add_options()
      ("games-dir,G", po::value<std::string>(&args.games_dir_str)->default_value("c4_games"), "where to write games")
      ;

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  TrainingDataWriter writer(args.games_dir_str);

  ParallelGameRunner runner;
  runner.register_players([&]() { return create_players(&writer); });
  runner.run();

  printf("\nWrote data to: %s\n", args.games_dir_str.c_str());
  return 0;
}
