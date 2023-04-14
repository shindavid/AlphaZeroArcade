#include <common/AbstractPlayer.hpp>

namespace common {

template<GameStateConcept GameState>
void AbstractPlayer<GameState>::init_game(
    game_id_t game_id, const player_name_array_t& player_names, seat_index_t seat_assignment)
{
  game_id_ = game_id;
  player_names_ = player_names;
  my_seat_ = seat_assignment;
}

}  // namespace common
