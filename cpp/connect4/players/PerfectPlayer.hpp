#pragma once

#include <mutex>

#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <boost/program_options.hpp>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <connect4/Constants.hpp>
#include <connect4/GameState.hpp>
#include <util/BoostUtil.hpp>

namespace c4 {

class PerfectOracle {
public:
  using GameStateTypes = common::GameStateTypes<c4::GameState>;
  using ActionMask = GameStateTypes::ActionMask;
  using ScoreArray = Eigen::Array<int, kNumColumns, 1>;

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
   * Interpretation of s = scores[k]:
   *
   * If s == kInvalidScore, then the move is not valid.
   *
   * If s < 0, then the move loses in -s moves against perfect counter-play.
   *
   * If s == 0, then the move results in a draw against perfect counter-play.
   *
   * If s > 0, then the move wins in s moves against perfect counter-play.
   *
   * If the position is winning, the member best_score is set to the positive score closest to zero. If the position is
   * losing, the member best_score is set to the negative score closest to zero. If the position is drawn, the member
   * best_score is set to 0.
   */
  struct QueryResult {
    static constexpr int kIllegalMoveScore = -1000;

    ScoreArray scores;
    int best_score;

    std::string get_overlay() const;
  };

  PerfectOracle();
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

  struct Params {
    /*
     * The strength parameter controls how well the player plays. It effectively acts as a look-ahead depth. More
     * specifically, the agent will choose randomly among all moves that can force a win within <strength> moves, if
     * such moves exist; otherwise, it will choose randomly among all moves that can avoid a loss within <strength>
     * moves, if such moves exist.
     *
     * if it can
     * force a win within <strength> moves, or avoid a loss within <strength> moves, it will do so. Ote
     *
     * Plays best-moves in mate/draw-in-N situations for N <= strength.
     *
     * In mate/draw-in-N situations for N > strength, randomly mixes in a second-best-move.
     *
     * In mate
     *
     * Randomly picks (21 - strength) integers among the 21 integers {1, 2, ..., 21}. The moves corresponding to
     * these integers are played weakly, by adding a non-best move to the set of candidate moves. The other moves
     * are played perfectly.
     *
     * A weak move is played by adding a single second-best move (as judged by the oracle) to the set of candidate
     * moves.
     */
    int strength = 21;
    bool verbose = false;

    auto make_options_description();
  };

  PerfectPlayer(const Params&);

  void start_game() override;
  void receive_state_change(common::seat_index_t, const GameState&, common::action_index_t) override;
  common::action_index_t get_action(const GameState&, const ActionMask&) override;

private:
  const Params params_;

  PerfectOracle oracle_;
  PerfectOracle::MoveHistory move_history_;
};

}  // namespace c4

#include <connect4/players/inl/PerfectPlayer.inl>
