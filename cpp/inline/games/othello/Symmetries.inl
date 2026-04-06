#include "games/othello/Symmetries.hpp"

#include "core/DefaultCanonicalizer.hpp"
#include "games/othello/Game.hpp"
#include "util/BitMapUtil.hpp"
#include "util/EigenUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/FiniteGroups.hpp"

#include <boost/lexical_cast.hpp>

namespace othello {

inline Game::Types::SymmetryMask Symmetries::get_mask(const Game::State& state) {
  Game::Types::SymmetryMask mask;
  mask.set();
  return mask;
}

inline void Symmetries::apply(Game::State& state, group::element_t sym) {
  using namespace bitmap_util;
  using D4 = groups::D4;
  auto& s = state;
  switch (sym) {
    case D4::kIdentity:
      return;
    case D4::kRot90:
      return rot90_clockwise(s.core.cur_player_mask, s.core.opponent_mask, s.aux.stable_discs);
    case D4::kRot180:
      return rot180(s.core.cur_player_mask, s.core.opponent_mask, s.aux.stable_discs);
    case D4::kRot270:
      return rot270_clockwise(s.core.cur_player_mask, s.core.opponent_mask, s.aux.stable_discs);
    case D4::kFlipVertical:
      return flip_vertical(s.core.cur_player_mask, s.core.opponent_mask, s.aux.stable_discs);
    case D4::kFlipMainDiag:
      return flip_main_diag(s.core.cur_player_mask, s.core.opponent_mask, s.aux.stable_discs);
    case D4::kMirrorHorizontal:
      return mirror_horizontal(s.core.cur_player_mask, s.core.opponent_mask, s.aux.stable_discs);
    case D4::kFlipAntiDiag:
      return flip_anti_diag(s.core.cur_player_mask, s.core.opponent_mask, s.aux.stable_discs);
    default:
      throw util::Exception("Unknown group element: {}", sym);
  }
}

template <eigen_util::concepts::FTensor Tensor>
inline void Symmetries::apply(Tensor& tensor, group::element_t sym, core::game_phase_t) {
  using namespace eigen_util;
  using D4 = groups::D4;
  constexpr int N = kBoardDimension;
  switch (sym) {
    case D4::kIdentity:
      return;
    case D4::kRot90:
      return rot90_clockwise<N>(tensor);
    case D4::kRot180:
      return rot180<N>(tensor);
    case D4::kRot270:
      return rot270_clockwise<N>(tensor);
    case D4::kFlipVertical:
      return flip_vertical<N>(tensor);
    case D4::kFlipMainDiag:
      return flip_main_diag<N>(tensor);
    case D4::kMirrorHorizontal:
      return mirror_horizontal<N>(tensor);
    case D4::kFlipAntiDiag:
      return flip_anti_diag<N>(tensor);
    default:
      throw util::Exception("Unknown group element: {}", sym);
  }
}

inline group::element_t Symmetries::get_canonical_symmetry(const Game::State& state) {
  using DefaultCanonicalizer = core::DefaultCanonicalizer<InputFrame, Symmetries>;
  return DefaultCanonicalizer::get(state);
}

}  // namespace othello
