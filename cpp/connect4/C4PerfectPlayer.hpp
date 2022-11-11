#pragma once

#include <boost/filesystem.hpp>
#include <boost/process.hpp>

#include <common/AbstractPlayer.hpp>
#include <common/Types.hpp>
#include <connect4/C4Constants.hpp>
#include <connect4/C4GameState.hpp>

namespace c4 {

class PerfectOracle {
public:
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

  PerfectOracle(const boost::filesystem::path& c4_solver_dir);
  ~PerfectOracle();
  QueryResult get_best_moves(MoveHistory& history, bool strong=true);
  static boost::filesystem::path get_default_c4_solver_dir();

private:
  boost::process::ipstream out_;
  boost::process::opstream in_;
  boost::process::child* proc_ = nullptr;
};

class PerfectPlayer : public Player {
public:
  using base_t = Player;

  struct Params {
    Params(const boost::filesystem::path& c, bool s=true) : c4_solver_dir(c), strong_mode(s) {}
    Params() : c4_solver_dir(PerfectOracle::get_default_c4_solver_dir()) {}

    boost::filesystem::path c4_solver_dir;
    bool strong_mode = true;
  };

  PerfectPlayer(const Params&);

  void start_game(const player_array_t& players, common::player_index_t seat_assignment) override;
  void receive_state_change(common::player_index_t, const GameState&, common::action_index_t, const Result&) override;
  common::action_index_t get_action(const GameState&, const ActionMask&) override;

private:
  PerfectOracle oracle_;
  PerfectOracle::MoveHistory move_history_;
  const bool strong_mode_;
};

}  // namespace c4

#include <connect4/C4PerfectPlayerINLINES.cpp>
