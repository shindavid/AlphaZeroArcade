#include "core/NetworkHeads.hpp"

#include "util/EigenUtil.hpp"

namespace core {

template <core::concepts::TensorEncodings TensorEncodings>
template <typename Derived>
void PolicyNetworkHead<TensorEncodings>::transform(Eigen::TensorBase<Derived>& dst) {
  eigen_util::softmax_in_place(dst);
}

template <core::concepts::TensorEncodings TensorEncodings>
template <typename Derived>
void PolicyNetworkHead<TensorEncodings>::uniform_init(Eigen::TensorBase<Derived>& dst) {
  dst.setConstant(1.0 / eigen_util::size(dst));
}

template <core::concepts::TensorEncodings TensorEncodings>
template <typename Derived>
void ValueNetworkHead<TensorEncodings>::transform(Eigen::TensorBase<Derived>& dst) {
  eigen_util::softmax_in_place(dst);
}

template <core::concepts::TensorEncodings TensorEncodings>
template <typename Derived>
void ValueNetworkHead<TensorEncodings>::uniform_init(Eigen::TensorBase<Derived>& dst) {
  dst.setConstant(1.0 / eigen_util::size(dst));
}

template <core::concepts::Game Game>
template <typename Derived>
void ActionValueNetworkHead<Game>::transform(Eigen::TensorBase<Derived>& dst) {
  eigen_util::rowwise_softmax_in_place(dst);
}

template <core::concepts::Game Game>
template <typename Derived>
void ActionValueNetworkHead<Game>::uniform_init(Eigen::TensorBase<Derived>& dst) {
  dst.setConstant(1.0 / Game::Constants::kNumPlayers);
}

template <core::concepts::Game Game>
template <typename Derived>
void ValueUncertaintyNetworkHead<Game>::uniform_init(Eigen::TensorBase<Derived>& dst) {
  dst.setConstant(0.5f);
}

template <core::concepts::Game Game>
template <typename Derived>
void ActionValueUncertaintyNetworkHead<Game>::uniform_init(Eigen::TensorBase<Derived>& dst) {
  dst.setConstant(0.5f);
}

}  // namespace core
