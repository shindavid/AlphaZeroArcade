#include "games/connect4/Game.hpp"

#include <boost/lexical_cast.hpp>

namespace c4 {

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const State& state) {
  mask_t bottomed_full_mask = state.full_mask + GameState::full_bottom_mask();

  Types::ActionMask mask;
  for (int col = 0; col < kNumColumns; ++col) {
    bool legal = bottomed_full_mask & GameState::column_mask(col);
    mask[col] = legal;
  }
  return mask;
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.get_current_player();
}

inline void Game::Rules::apply(State& state, core::action_t action) {
  column_t col = action;
  auto bottom_mask = GameState::bottom_mask(col);
  auto column_mask = GameState::column_mask(col);
  mask_t piece_mask = (state.full_mask + bottom_mask) & column_mask;

  state.cur_player_mask ^= state.full_mask;
  state.full_mask |= piece_mask;
  state.last_action = action;
}

inline void Game::IO::add_render_info(const State& state, boost::json::object& msg) {
  boost::json::array col_heights;
  for (int col = 0; col < kNumColumns; ++col) {
    col_heights.push_back(kNumRows - state.num_empty_cells(col));
  }
  msg["col_heights"] = col_heights;
}

}  // namespace c4
