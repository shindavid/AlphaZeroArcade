#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "core/OraclePool.hpp"
#include "games/connect4/Constants.hpp"
#include "games/connect4/Game.hpp"
#include "games/connect4/PerfectOracle.hpp"
#include "util/BoostUtil.hpp"

namespace c4 {

class PerfectPlayer : public core::AbstractPlayer<c4::Game> {
 public:
  using base_t = core::AbstractPlayer<c4::Game>;
  using OraclePool = core::OraclePool<PerfectOracle>;

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

    // The number of oracle processes to use. If not specified, defaults to the number of game
    // threads.
    int num_oracle_procs = 0;

    bool verbose = false;

    auto make_options_description();
  };

  PerfectPlayer(OraclePool* oracle_pool, const Params&);

  void start_game() override;
  void receive_state_change(core::seat_index_t, const State&, core::action_t) override;
  ActionResponse get_action_response(const ActionRequest& request) override;

 private:
  OraclePool* const oracle_pool_;
  const Params params_;

  PerfectOracle::MoveHistory move_history_;
};

}  // namespace c4

#include "inline/games/connect4/players/PerfectPlayer.inl"
