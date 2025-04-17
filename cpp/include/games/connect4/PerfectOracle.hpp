#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/HibernationNotifier.hpp>
#include <games/connect4/Constants.hpp>
#include <games/connect4/Game.hpp>
#include <util/Asserts.hpp>
#include <util/BoostUtil.hpp>

#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <boost/program_options.hpp>

#include <mutex>
#include <string>
#include <vector>

namespace c4 {

class PerfectOracle {
 public:
  using ActionMask = Game::Types::ActionMask;
  using State = Game::State;
  using ScoreArray = Eigen::Array<int, kNumColumns, 1>;

  class MoveHistory {
   public:
    MoveHistory();
    MoveHistory(const MoveHistory&);

    void reset();
    void append(int move);
    std::string to_string() const;
    int length() const { return char_pointer_ - chars_; }

   private:
    void write(boost::process::opstream& in) const;

    mutable char chars_[kMaxMovesPerGame + 1];
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

  // Query the oracle. Blocks until the oracle has finished processing the query.
  QueryResult query(const MoveHistory& history);

  PerfectOracle();
  ~PerfectOracle();

 private:
  friend class PerfectOraclePool;

  boost::process::ipstream out_pipe_;
  boost::process::opstream in_pipe_;
  boost::process::child child_;

  mutable std::mutex mutex_;
};

class PerfectOraclePool {
 public:
  // capacity = max number of oracles instances to create
  PerfectOraclePool(int capacity = 16) { set_capacity(capacity); }

  void set_capacity(int capacity);

  // If there are fewer than capacity_ busy oracles, then returns a free oracle (creating a new one
  // if necessary). Otherwise, returns nullptr, and schedules the passed-in notifier to be
  // notified when an oracle becomes available.
  PerfectOracle* get_oracle(core::HibernationNotifier* notifier=nullptr);

  void release_oracle(PerfectOracle* oracle);

 private:
  using oracle_vec_t = std::vector<PerfectOracle*>;
  using notifier_vec_t = std::vector<core::HibernationNotifier*>;

  oracle_vec_t free_oracles_;
  notifier_vec_t pending_notifiers_;
  mutable std::mutex mutex_;
  int capacity_;
  int count_ = 0;
};

}  // namespace c4

#include <inline/games/connect4/PerfectOracle.inl>
