#include <games/tictactoe/GameState.hpp>

#include <core/SquareBoardSymmetries.hpp>

inline std::size_t std::hash<tictactoe::GameState>::operator()(
    const tictactoe::GameState& state) const {
  return state.hash();
}

namespace tictactoe {

inline GameState::SymmetryIndexSet GameState::get_symmetry_indices() const {
  SymmetryIndexSet set;
  set.set();
  return set;
}

inline core::seat_index_t GameState::get_current_player() const {
  return std::popcount(full_mask_) % 2;
}

inline core::seat_index_t GameState::get_player_at(int row, int col) const {
  int cp = get_current_player();
  int index = row * kBoardDimension + col;
  bool occupied_by_cur_player = (mask_t(1) << index) & cur_player_mask_;
  bool occupied_by_any_player = (mask_t(1) << index) & full_mask_;
  return occupied_by_any_player ? (occupied_by_cur_player ? cp : (1 - cp)) : -1;
}

}  // namespace tictactoe
