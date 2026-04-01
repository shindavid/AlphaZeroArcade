#include "games/hex/Symmetries.hpp"

#include "core/DefaultCanonicalizer.hpp"
#include "util/Exceptions.hpp"

namespace hex {

inline Game::Types::SymmetryMask Symmetries::get_mask(const InputFrame& frame) {
  return Game::Types::SymmetryMask().set();
}

inline void Symmetries::apply(InputFrame& frame, group::element_t sym) {
  switch (sym) {
    case groups::C2::kIdentity:
      return;
    case groups::C2::kRot180:
      return frame.rotate();
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

inline group::element_t Symmetries::get_canonical_symmetry(const InputFrame& frame) {
  using DefaultCanonicalizer = core::DefaultCanonicalizer<InputFrame, Symmetries>;
  return DefaultCanonicalizer::get(frame);
}

}  // namespace hex
