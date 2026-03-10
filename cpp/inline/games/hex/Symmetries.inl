#include "games/hex/Symmetries.hpp"

#include "core/DefaultCanonicalizer.hpp"
#include "util/Exceptions.hpp"

namespace hex {

inline Game::Types::SymmetryMask Symmetries::get_mask(const Game::State& state) {
  return Game::Types::SymmetryMask().set();
}

inline void Symmetries::apply(Game::State& state, group::element_t sym) {
  switch (sym) {
    case groups::C2::kIdentity:
      return;
    case groups::C2::kRot180:
      return state.rotate();
    default:
      throw util::Exception("Unknown group element: {}", sym);
  }
}

template <eigen_util::concepts::FTensor Tensor>
void Symmetries::apply(Tensor& tensor, group::element_t sym, core::action_mode_t) {
  constexpr int N = Constants::kBoardDim;
  switch (sym) {
    case groups::C2::kIdentity:
      return;
    case groups::C2::kRot180:
      return eigen_util::rot180<N>(tensor);
    default:
      throw util::Exception("Unknown group element: {}", sym);
  }
}

inline group::element_t Symmetries::get_canonical_symmetry(const Game::State& state) {
  using DefaultCanonicalizer = core::DefaultCanonicalizer<Game, Symmetries>;
  return DefaultCanonicalizer::get(state);
}

}  // namespace hex
