#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <games/connect4/Constants.hpp>
#include <games/connect4/Game.hpp>
#include <util/BoostUtil.hpp>

#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <boost/program_options.hpp>

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

}  // namespace c4

#include <inline/games/connect4/PerfectOracle.inl>
