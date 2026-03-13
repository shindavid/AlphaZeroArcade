#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/ActionRequest.hpp"
#include "core/concepts/GameConcept.hpp"

#include <random>

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

  RandomPlayer(int base_seed = -1) : base_seed_(base_seed) {}

  bool start_game() override;
  core::ActionResponse get_action_response(const ActionRequest& request) override;
  int base_seed() const { return base_seed_; }

 private:
  int base_seed_;
  std::mt19937 prng_;
};

}  // namespace generic

#include "inline/generic_players/RandomPlayer.inl"
