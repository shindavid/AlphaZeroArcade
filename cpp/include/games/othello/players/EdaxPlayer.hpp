#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/ActionResponse.hpp"
#include "core/OraclePool.hpp"
#include "games/othello/EdaxOracle.hpp"
#include "games/othello/Game.hpp"

#include <vector>

namespace othello {

/*
 * EdaxPlayer is a player that uses the edax engine to play Othello.
 *
 * See: https://github.com/okuhara/edax-reversi-AVX
 *
 * Currently, we interact with the edax engine in a clunky, inefficient way: we launch the edax
 * process, submit moves to it via stdin, and parse the stdout to get the engine's response. Later,
 * we can try to avoid the I/O and the text parsing by building and linking against an edax library
 * directly.
 */
class EdaxPlayer : public core::AbstractPlayer<Game> {
 public:
  using OraclePool = core::OraclePool<EdaxOracle>;
  using Move = Game::Move;
  using ActionResponse = core::ActionResponse<Game>;

  struct Params {
    int depth = 21;  // matches edax default

    // The number of oracle processes to use. If not specified, defaults to the number of game
    // threads.
    int num_oracle_procs = 0;

    bool deterministic = false;
    bool verbose = false;

    auto make_options_description();
  };

  EdaxPlayer(OraclePool* oracle_pool, const Params&);

  void end_game(const State& state, const GameResultTensor& results) override;
  ActionResponse get_action_response(const ActionRequest& request) override;

 private:
  OraclePool* const oracle_pool_;
  const Params params_;
  std::vector<Move*> aux_data_ptrs_;
};

}  // namespace othello

#include "inline/games/othello/players/EdaxPlayer.inl"
