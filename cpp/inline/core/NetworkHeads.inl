#include "core/NetworkHeads.hpp"

namespace core {

template <core::concepts::Game Game>
template <typename Derived>
void PolicyNetworkHead<Game>::transform(Eigen::TensorBase<Derived>& dst) {
  eigen_util::softmax_in_place(dst);
}

template <core::concepts::Game Game>
template <typename Derived>
void PolicyNetworkHead<Game>::uniform_init(Eigen::TensorBase<Derived>& dst) {
  dst.setConstant(1.0 / eigen_util::size(dst));
}

template <core::concepts::Game Game>
template <typename Derived>
void ValueNetworkHead<Game>::transform(Eigen::TensorBase<Derived>& dst) {
  eigen_util::softmax_in_place(dst);
}

template <core::concepts::Game Game>
template <typename Derived>
void ValueNetworkHead<Game>::uniform_init(Eigen::TensorBase<Derived>& dst) {
  dst.setConstant(1.0 / eigen_util::size(dst));
}

template <core::concepts::Game Game>
template <typename Derived>
void ActionValueNetworkHead<Game>::transform(Eigen::TensorBase<Derived>& dst) {
  eigen_util::sigmoid_in_place(dst);
}

template <core::concepts::Game Game>
template <typename Derived>
void ActionValueNetworkHead<Game>::uniform_init(Eigen::TensorBase<Derived>& dst) {
  dst.setConstant(1.0 / Game::Constants::kNumPlayers);
}

template <core::concepts::Game Game>
template <typename Derived>
void ValueUncertaintyNetworkHead<Game>::transform(Eigen::TensorBase<Derived>& dst) {
  eigen_util::sigmoid_in_place(dst);
}

template <core::concepts::Game Game>
template <typename Derived>
void ValueUncertaintyNetworkHead<Game>::uniform_init(Eigen::TensorBase<Derived>& dst) {
  constexpr float p = 1.0f / Game::Constants::kNumPlayers;
  dst.setConstant(p * (1 - p));
}

template <core::concepts::Game Game>
template <typename Derived>
void ActionValueUncertaintyNetworkHead<Game>::transform(Eigen::TensorBase<Derived>& dst) {
  eigen_util::sigmoid_in_place(dst);
}

template <core::concepts::Game Game>
template <typename Derived>
void ActionValueUncertaintyNetworkHead<Game>::uniform_init(Eigen::TensorBase<Derived>& dst) {
  constexpr float p = 1.0f / Game::Constants::kNumPlayers;
  dst.setConstant(p * (1 - p));
}

}  // namespace core
