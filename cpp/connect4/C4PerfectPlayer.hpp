#pragma once

#include <mutex>

#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <boost/program_options.hpp>

#include <common/AbstractPlayer.hpp>
#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameRunner.hpp>
#include <connect4/C4Constants.hpp>
#include <connect4/C4GameState.hpp>
#include <util/BoostUtil.hpp>

namespace c4 {

struct PerfectPlayParams {
  std::string c4_solver_dir;

  /*
   * In leisurely mode, PerfectPlayer has no preference among winning moves.
   *
   * Else, it prefers the fastest win.
   *
   * In losing positions, PerfectPlayer prefers the slowest loss regardless of leisurely_mode.
   */
  bool leisurely_mode = false;

  auto make_options_description();
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
    std::string to_string() const;

  private:
    void write(boost::process::opstream& in);

    char chars_[kMaxMovesPerGame + 1];
    char* char_pointer_;
    friend class PerfectOracle;
  };

  /*
   * If current player wins with optimal play...
   *  - score will be > 0
   *  - good_moves will be a mask of all moves that win against optimal play
   *  - best_moves will be a mask of all moves leading to the fastest win against optimal play
   *
   * If current player loses against optimal play...
   *  - score will be < 0
   *  - good_moves will be a mask of all valid moves
   *  - best_moves will be a mask of all moves leading to the slowest loss against optimal play
   *
   * If current player can force a draw against optimal play...
   *  - good_moves and best_moves will be a mask of all moves forcing a draw
   *  - score will be 0
   */
  struct QueryResult {
    ActionMask best_moves;
    ActionMask good_moves;
    int score;
  };

  PerfectOracle(const PerfectPlayParams& params);
  ~PerfectOracle();

  QueryResult query(MoveHistory& history);

private:
  std::mutex mutex_;
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
  const bool leisurely_mode_;
};

class PerfectGrader {
public:
  struct stats_t {
    int correct_count = 0;
    int total_count = 0;

    void update(bool correct);
    stats_t& operator+=(const stats_t& rhs);
  };

  /*
   * move_number_t:
   *
   * - k>0 signifies the k'th move of the game
   * - k<0 signifies the k'th move of the game, counting from the end
   * - k==0 signifies all moves
   */
  using move_number_t = int;
  using key_t = std::tuple<common::player_index_t, move_number_t>;  // player_index_t of -1 means all players
  using stats_map_t = std::map<key_t, stats_t>;

  class Listener : public common::GameRunner<c4::GameState>::Listener {
  public:
    Listener(PerfectGrader& grader) : grader_(grader) {}
    void on_game_start(common::game_id_t) override;
    void on_game_end() override;
    void on_move(common::player_index_t, common::action_index_t) override;

  private:
    PerfectGrader& grader_;
    PerfectOracle::MoveHistory move_history_;
    move_number_t move_number_ = 0;
    stats_map_t tmp_stats_map_;  // tracks only for current game
  };

  PerfectGrader(const PerfectPlayParams& params) : oracle_(params) {}

  Listener* make_listener() { return new Listener(*this); }
  void dump() const;
  PerfectOracle& oracle() { return oracle_; }
  stats_map_t& stats_map() { return stats_map_; }

private:
  PerfectOracle oracle_;
  stats_map_t stats_map_;  // tracks over all games
};

}  // namespace c4

#include <connect4/inl/C4PerfectPlayer.inl>
