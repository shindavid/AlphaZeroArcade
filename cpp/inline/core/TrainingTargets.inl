#include <core/TrainingTargets.hpp>

#include <core/BasicTypes.hpp>

namespace core {

template<typename Game>
typename PolicyTarget<Game>::Tensor
PolicyTarget<Game>::tensorize(const GameLogView& view) {
  return *view.policy;
}

template <typename Game>
typename WinLossDrawTarget<Game>::Tensor WinLossDrawTarget<Game>::tensorize(
    const GameLogView& view) {
  using Rules = Game::Rules;
  seat_index_t cp = Rules::get_current_player(*view.cur_pos);
  Tensor tensor = *view.game_result;
  Game::GameResults::left_rotate(tensor, cp);
  return tensor;
}

template <typename Game>
typename ActionValueTarget<Game>::Tensor ActionValueTarget<Game>::tensorize(
    const GameLogView& view) {
  // GameLogView has action_values, which is of type ActionValueTensor. We need to transform this
  // to FullActionValueTensor.
  //
  // The difference is that FullActionValueTensor has one extra plane for the invalid action.
  //
  // To perform this translation, we have to look at each slice. If the slice is all zeros, then
  // we need to set the corresponding entry in the invalid action plane to 1.
  //
  // TODO: considering aligning to current-player before writing to disk. Then we don't need to
  // align on the fly here.
  using FullShape = Tensor::Dimensions;
  static_assert(FullShape::count == 2);
  constexpr int N = eigen_util::extract_dim_v<0, FullShape>;

  Tensor tensor;

  using StartIndices = Eigen::array<Eigen::Index, 1>;
  using Sizes = Eigen::array<Eigen::Index, 1>;

  seat_index_t cp = Game::Rules::get_current_player(*view.cur_pos);

  int j = 0;
  eigen_util::compute_per_slice<1>(*view.action_values, [&](const auto& slice) {
    using SliceT = std::decay_t<decltype(slice)>;
    using Shape = SliceT::Dimensions;
    static_assert(eigen_util::extract_dim_v<0, Shape> == N - 1);

    SliceT rotated_slice = slice;
    Game::GameResults::left_rotate(rotated_slice, cp);
    tensor.chip(j, 1).slice(StartIndices{0}, Sizes{N - 1}) = rotated_slice;
    tensor(N - 1, j) = eigen_util::any(slice) ? 0 : 1;
    j++;
  });

  return tensor;
}

template<typename Game>
typename OppPolicyTarget<Game>::Tensor
OppPolicyTarget<Game>::tensorize(const GameLogView& view) {
  return *view.next_policy;
}

}  // namespace core
