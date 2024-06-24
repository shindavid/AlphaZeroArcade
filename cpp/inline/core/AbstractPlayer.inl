#include <core/AbstractPlayer.hpp>

namespace core {

template <concepts::Game Game>
void AbstractPlayer<Game>::init_game(game_id_t game_id,
                                     const player_name_array_t& player_names,
                                     seat_index_t seat_assignment) {
  game_id_ = game_id;
  player_names_ = player_names;
  my_seat_ = seat_assignment;
}

}  // namespace core
