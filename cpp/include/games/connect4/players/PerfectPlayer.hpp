#pragma once

#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <boost/program_options.hpp>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <util/BoostUtil.hpp>

#include <games/connect4/Constants.hpp>
#include <games/connect4/Game.hpp>

#include <mutex>
#include <vector>

namespace c4 {

class PerfectOracle {
 public:
  using ActionMask = Game::Types::ActionMask;
  using State = Game::State;
  using ScoreArray = Eigen::Array<int, kNumColumns, 1>;

  static constexpr int kNumClientsPerOracle = 16;
  using oracle_vec_t = std::vector<PerfectOracle*>;

  class MoveHistory {
   public:
    MoveHistory();
    MoveHistory(const MoveHistory&);

    void reset();
    void append(int move);
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
   * If the position is winning, best_score is set to the positive score closest to zero.
   * If the position is losing, best_score is set to the negative score furthest to zero.
   * If the position is drawn, best_score is set to 0.
   */
  struct QueryResult {
    static constexpr int kIllegalMoveScore = -1000;

    ScoreArray scores;
    int best_score;

    std::string get_overlay() const;
  };

  static PerfectOracle* get_instance();

  QueryResult query(MoveHistory& history);

 private:
  PerfectOracle();
  ~PerfectOracle();

  static oracle_vec_t oracles_;
  static std::mutex static_mutex_;

  std::mutex mutex_;
  boost::process::ipstream out_;
  boost::process::opstream in_;
  boost::process::child* proc_ = nullptr;
  int client_count_ = 0;
};

class PerfectPlayer : public core::AbstractPlayer<c4::Game> {
 public:
  using base_t = core::AbstractPlayer<c4::Game>;

  struct Params {
    /*
     * The strength parameter controls how well the player plays. It effectively acts as a
     * look-ahead depth. More specifically, the agent will choose randomly among all moves that can
     * force a win within <strength> moves, if such moves exist; otherwise, it will choose randomly
     * among all moves that can avoid a loss within <strength> moves, if such moves exist.
     *
     * When the agent knows that is it losing, it will choose randomly among all moves that can
     * delay the loss the longest.
     *
     * NOTE[dshin]: I experimented with changing the behavior of the agent when it knows it is
     * losing. Instead of choosing uniformly randomly among the slowest losses, I tried something
     * that will yield a little more variety: choose among all actions, with a probability
     * proportional to the 2^k, where k is the number of moves it takes to lose against optimal
     * play. This is a little more interesting, but empirically, it makes the agent clearly
     * weaker against imperfect MCTS agents. So ultimately I decided to stick with the uniform
     * random choice among the slowest losses, to make the agent as strong as possible.
     */
    int strength = 21;
    bool verbose = false;

    auto make_options_description();
  };

  PerfectPlayer(const Params&);

  void start_game() override;
  void receive_state_change(core::seat_index_t, const State&, core::action_t) override;
  ActionResponse get_action_response(const ActionRequest& request) override;

 private:
  const Params params_;

  PerfectOracle* oracle_;
  PerfectOracle::MoveHistory move_history_;
};

}  // namespace c4

#include <inline/games/connect4/players/PerfectPlayer.inl>
