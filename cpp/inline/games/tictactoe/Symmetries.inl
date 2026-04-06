#include "games/tictactoe/Symmetries.hpp"

#include "core/DefaultCanonicalizer.hpp"

namespace tictactoe {

namespace detail {

inline void flip_vertical(mask_t& mask) {
  mask = ((mask & 0x07) << 6) | ((mask & 0x38) << 0) | ((mask & 0x1c0) >> 6);
}

inline void flip_main_diag(mask_t& mask) {
  mask = (mask & 0x111) | ((mask & 0x22) << 2) | ((mask & 0x4) << 4) | ((mask & 0x88) >> 2) |
         ((mask & 0x40) >> 4);
}

inline void mirror_horizontal(mask_t& mask) {
  mask = ((mask & 0x49) << 2) | ((mask & 0x92) << 0) | ((mask & 0x124) >> 2);
}

inline void flip_anti_diag(mask_t& mask) {
  mask = (mask & 0x54) | ((mask & 0x100) >> 8) | ((mask & 0x01) << 8) | ((mask & 0x0a) << 4) |
         ((mask & 0xa0) >> 4);
}

inline void rot90_clockwise(mask_t& mask) {
  flip_vertical(mask);
  flip_main_diag(mask);
}

inline void rot180(mask_t& mask) {
  flip_vertical(mask);
  mirror_horizontal(mask);
}

inline void rot270_clockwise(mask_t& mask) {
  flip_vertical(mask);
  flip_anti_diag(mask);
}

inline void rot90_clockwise(mask_t& mask1, mask_t& mask2) {
  rot90_clockwise(mask1);
  rot90_clockwise(mask2);
}

inline void rot180(mask_t& mask1, mask_t& mask2) {
  rot180(mask1);
  rot180(mask2);
}

inline void rot270_clockwise(mask_t& mask1, mask_t& mask2) {
  rot270_clockwise(mask1);
  rot270_clockwise(mask2);
}

inline void flip_vertical(mask_t& mask1, mask_t& mask2) {
  flip_vertical(mask1);
  flip_vertical(mask2);
}

inline void flip_main_diag(mask_t& mask1, mask_t& mask2) {
  flip_main_diag(mask1);
  flip_main_diag(mask2);
}

inline void mirror_horizontal(mask_t& mask1, mask_t& mask2) {
  mirror_horizontal(mask1);
  mirror_horizontal(mask2);
}

inline void flip_anti_diag(mask_t& mask1, mask_t& mask2) {
  flip_anti_diag(mask1);
  flip_anti_diag(mask2);
}

}  // namespace detail

inline Game::Types::SymmetryMask Symmetries::get_mask(const Game::State& state) {
  Game::Types::SymmetryMask mask;
  mask.set();
  return mask;
}

inline void Symmetries::apply(Game::State& state, group::element_t sym) {
  using namespace tictactoe::detail;
  using D4 = groups::D4;
  auto& s = state;
  switch (sym) {
    case D4::kIdentity:
      return;
    case D4::kRot90:
      return rot90_clockwise(s.cur_player_mask, s.full_mask);
    case D4::kRot180:
      return rot180(s.cur_player_mask, s.full_mask);
    case D4::kRot270:
      return rot270_clockwise(s.cur_player_mask, s.full_mask);
    case D4::kFlipVertical:
      return flip_vertical(s.cur_player_mask, s.full_mask);
    case D4::kFlipMainDiag:
      return flip_main_diag(s.cur_player_mask, s.full_mask);
    case D4::kMirrorHorizontal:
      return mirror_horizontal(s.cur_player_mask, s.full_mask);
    case D4::kFlipAntiDiag:
      return flip_anti_diag(s.cur_player_mask, s.full_mask);
    default: {
      throw util::Exception("Unknown group element: {}", sym);
    }
  }
}

template <eigen_util::concepts::FTensor Tensor>
void Symmetries::apply(Tensor& tensor, group::element_t sym, const InputFrame&) {
  apply(tensor, sym);
}

template <eigen_util::concepts::FTensor Tensor>
void Symmetries::apply(Tensor& tensor, group::element_t sym) {
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
    default: {
      throw util::Exception("Unknown group element: {}", sym);
    }
  }
}

inline group::element_t Symmetries::get_canonical_symmetry(const Game::State& state) {
  using DefaultCanonicalizer = core::DefaultCanonicalizer<InputFrame, Symmetries>;
  return DefaultCanonicalizer::get(state);
}

}  // namespace tictactoe
