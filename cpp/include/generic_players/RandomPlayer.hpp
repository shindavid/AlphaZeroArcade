#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/ActionRequest.hpp"
#include "core/ActionResponse.hpp"
#include "core/concepts/GameConcept.hpp"

#include <random>
#include <vector>

namespace generic {

/*
 * RandomPlayer always chooses uniformly at random among the set of legal moves.
 *
 * If base_seed >= 0, the PRNG is reseeded at the start of each game using (base_seed + game_id),
 * making the action sequence deterministic per game regardless of thread scheduling. Otherwise,
 * falls back to the global default PRNG.
 */
template <core::concepts::Game Game>
class RandomPlayer : public core::AbstractPlayer<Game> {
 public:
  using ActionRequest = core::ActionRequest<Game>;
  using ActionResponse = core::ActionResponse<Game>;
  using InfoSet = Game::InfoSet;
  using Move = Game::Move;
  using GameOutcome = Game::Types::GameOutcome;

  RandomPlayer(int base_seed = -1) : base_seed_(base_seed) {}

  bool start_game() override;
  void end_game(const InfoSet& state, const GameOutcome& results) override;
  ActionResponse get_action_response(const ActionRequest& request) override;
  int base_seed() const { return base_seed_; }

 private:
  std::vector<Move*> aux_data_ptrs_;
  int base_seed_;
  std::mt19937 prng_;
};

}  // namespace generic

#include "inline/generic_players/RandomPlayer.inl"
