#include "generic_players/RandomPlayer.hpp"

#include "util/Random.hpp"

namespace generic {

template <core::concepts::Game Game>
bool RandomPlayer<Game>::start_game() {
  if (base_seed_ >= 0) {
    prng_.seed(base_seed_ + this->get_game_id());
  }
  return true;
}

template <core::concepts::Game Game>
void RandomPlayer<Game>::end_game(const InfoSet& state, const GameOutcome& results) {
  for (auto ptr : aux_data_ptrs_) {
    delete ptr;
  }
  aux_data_ptrs_.clear();
}

template <core::concepts::Game Game>
typename RandomPlayer<Game>::ActionResponse RandomPlayer<Game>::get_action_response(
  const ActionRequest& request) {
  if (request.aux) {
    return *reinterpret_cast<Move*>(request.aux);
  }

  auto& prng = (base_seed_ >= 0) ? prng_ : util::Random::default_prng();
  Move move = request.valid_moves.get_random(prng);
  ActionResponse response(move);

  if (this->is_facing_backtracking_opponent()) {
    Move* move_ptr = new Move(move);
    aux_data_ptrs_.push_back(move_ptr);
    response.set_aux(move_ptr);
  }

  return response;
}

}  // namespace generic
