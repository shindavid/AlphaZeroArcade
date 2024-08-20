#include <games/blokus/Game.hpp>

#include <util/CppUtil.hpp>

namespace blokus {

inline size_t Game::BaseState::hash() const {
  return util::PODHash<core_t>{}(core);
}

inline Game::Types::SymmetryMask Game::Symmetries::get_mask(const BaseState& state) {
  Types::SymmetryMask mask;
  mask.set();
  return mask;
}

inline core::seat_index_t Game::Rules::get_current_player(const BaseState& state) {
  return state.core.cur_color;
}

/*
 * TODO: Currently, I'm overloading actions to be both the piece and the position, depending on
 * which submove is being made. This makes it impossible to write action_to_str().
 *
 * I probably want to change action_to_str() to accept the current state to resolve this.
 */
inline std::string Game::IO::action_to_str(core::action_t action) {
  throw std::runtime_error("Not implemented");
}

}  // namespace blokus
