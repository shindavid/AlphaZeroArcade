#pragma once

#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <boost/program_options.hpp>

#include <common/AbstractPlayer.hpp>
#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <connect4/C4Constants.hpp>
#include <connect4/C4GameState.hpp>

namespace c4 {

struct PerfectPlayParams {
  std::string c4_solver_dir;
  bool weak_mode = false;

  boost::program_options::options_description make_options_description(bool add_shortcuts=false);
};

class PerfectOracle {
public:
  using GameStateTypes = common::GameStateTypes<c4::GameState>;
  using ActionMask = GameStateTypes::ActionMask;

  class MoveHistory {
  public:
    MoveHistory();
    void reset();
    void append(common::action_index_t move);

  private:
    void write(boost::process::opstream& in);

    char chars_[kMaxMovesPerGame + 1];
    char* char_pointer_;
    friend class PerfectOracle;
  };

  struct QueryResult {
    ActionMask moves;
    int score;
  };

  PerfectOracle(const PerfectPlayParams& params);
  ~PerfectOracle();

  QueryResult get_best_moves(MoveHistory& history);

private:
  const bool weak_mode_;

  boost::process::ipstream out_;
  boost::process::opstream in_;
  boost::process::child* proc_ = nullptr;
};

class PerfectPlayer : public Player {
public:
  using base_t = Player;

  PerfectPlayer(const PerfectPlayParams&);

  void start_game(common::game_id_t, const player_array_t& players, common::player_index_t seat_assignment) override;
  void receive_state_change(common::player_index_t, const GameState&, common::action_index_t, const GameOutcome&) override;
  common::action_index_t get_action(const GameState&, const ActionMask&) override;

private:
  PerfectOracle oracle_;
  PerfectOracle::MoveHistory move_history_;
};

}  // namespace c4

#include <connect4/inl/C4PerfectPlayer.inl>
