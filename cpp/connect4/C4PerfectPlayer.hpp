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
    MoveHistory(const MoveHistory&);

    void reset();
    void append(common::action_index_t move);
    std::string to_string() const;
    int length() const { return char_pointer_ - chars_; }

  private:
    void write(boost::process::opstream& in);

    char chars_[kMaxMovesPerGame + 1];
    char* char_pointer_;
    friend class PerfectOracle;
  };

  /*
   * Let P be current player and Q be other player.
   *
   * If P wins with optimal play...
   *  - score will be N, where N is the number of additional moves P needs.
   *  - good_moves will be a mask of all moves that win against optimal play
   *  - best_moves will be a mask of all moves leading to the fastest win against optimal play
   *
   * If Q wins with optimal play...
   *  - score will be -N, where N is the number of additional moves Q needs
   *  - good_moves will be a mask of all valid moves
   *  - best_moves will be a mask of all moves leading to the slowest loss against optimal play
   *
   * If P can force a draw against optimal play...
   *  - good_moves and best_moves will be a mask of all moves forcing a draw
   *  - score will be 0
   *
   * For *exact* queries, the score, when positive, will correspond to the number of moves left in the
   * game given perfect play by both sides.
   */
  struct QueryResult {
    ActionMask best_moves;
    ActionMask good_moves;
    ActionMask drawing_moves;
    int score;

    std::string get_overlay() const;
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

}  // namespace c4

#include <connect4/inl/C4PerfectPlayer.inl>
