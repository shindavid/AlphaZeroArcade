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
core::ActionResponse RandomPlayer<Game>::get_action_response(const ActionRequest& request) {
  if (request.aux) {
    return request.aux - 1;
  }

  auto& prng = (base_seed_ >= 0) ? prng_ : util::Random::default_prng();
  core::ActionResponse response(request.valid_actions.choose_random_on_index(prng));
  response.set_aux(response.get_action() + 1);

  return response;
}

}  // namespace generic
