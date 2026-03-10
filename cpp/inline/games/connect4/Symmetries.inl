#include "games/connect4/Symmetries.hpp"

#include "core/BasicTypes.hpp"
#include "core/DefaultCanonicalizer.hpp"
#include "util/EigenUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/FiniteGroups.hpp"

#include <bit>

namespace c4 {

inline Game::Types::SymmetryMask Symmetries::get_mask(const Game::State& state) {
  Game::Types::SymmetryMask mask;
  mask.set();
  return mask;
}

inline void Symmetries::apply(Game::State& state, group::element_t sym) {
  switch (sym) {
    case groups::D1::kIdentity:
      return;
    case groups::D1::kFlip: {
      state.full_mask = std::byteswap(state.full_mask << 8);
      state.cur_player_mask = std::byteswap(state.cur_player_mask << 8);
      return;
    }
    default: {
      throw util::Exception("Unknown group element: {}", sym);
    }
  }
}

template <eigen_util::concepts::FTensor Tensor>
inline void Symmetries::apply(Tensor& t, group::element_t sym, core::action_mode_t) {
  switch (sym) {
    case groups::D1::kIdentity:
      return;
    case groups::D1::kFlip: {
      Tensor u = eigen_util::reverse(t, 0);
      t = u;
      return;
    }
    default: {
      throw util::Exception("Unknown group element: {}", sym);
    }
  }
}

inline group::element_t Symmetries::get_canonical_symmetry(const Game::State& state) {
  using DefaultCanonicalizer = core::DefaultCanonicalizer<Game, Symmetries>;
  return DefaultCanonicalizer::get(state);
}

}  // namespace c4
