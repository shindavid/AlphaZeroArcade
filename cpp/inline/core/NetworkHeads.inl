#include "core/NetworkHeads.hpp"

#include "util/EigenUtil.hpp"

namespace core {

namespace detail {

template <typename Tensor>
using TensorMap = Eigen::TensorMap<Tensor, Eigen::Aligned>;

template <typename Tensor>
using DynamicTensor = Eigen::Tensor<float, Tensor::NumIndices, Eigen::RowMajor>;

template <typename Tensor>
inline auto make_tensor_map(float* data) {
  return TensorMap<Tensor>(data, eigen_util::to_int64_std_array_v<typename Tensor::Dimensions>);
}

template <typename Tensor>
inline auto make_per_action_tensor_map(float* data, int num_valid_moves) {
  auto dims = eigen_util::to_int64_std_array_v<typename Tensor::Dimensions>;
  dims[0] = num_valid_moves;
  return TensorMap<DynamicTensor<Tensor>>(data, dims);
}

}  // namespace detail

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
template <typename InitParams>
void PolicyNetworkHead<TensorEncodings, Symmetries>::load(float* data, Tensor& src,
                                                         const InitParams& params) {
  group::element_t inv_sym = Game::SymmetryGroup::inverse(params.sym);
  Symmetries::apply(src, inv_sym, params.frame);

  auto dst = detail::make_per_action_tensor_map<Tensor>(data, params.valid_moves.size());
  int i = 0;
  for (Move move : params.valid_moves) {
    auto index = PolicyEncoding::to_index(params.frame, move);
    dst.chip(i++, 0) = eigen_util::chip_recursive(src, index);
  }

  eigen_util::softmax_in_place(dst);
}

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
int PolicyNetworkHead<TensorEncodings, Symmetries>::size(int num_valid_moves) {
  using Shape = Tensor::Dimensions;
  return (Tensor::Dimensions::total_size / eigen_util::extract_dim_v<0, Shape>)*num_valid_moves;
}

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
void PolicyNetworkHead<TensorEncodings, Symmetries>::uniform_init(float* data, int num_valid_moves) {
  auto dst = detail::make_per_action_tensor_map<Tensor>(data, num_valid_moves);
  dst.setConstant(1.0f / eigen_util::size(dst));
}

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
template <typename InitParams>
void ValueNetworkHead<TensorEncodings, Symmetries>::load(float* data, Tensor& src,
                                                        const InitParams& params) {
  GameResultEncoding::right_rotate(src, params.active_seat);
  auto dst = detail::make_tensor_map<Tensor>(data);
  dst = src;
  eigen_util::softmax_in_place(dst);
}

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
int ValueNetworkHead<TensorEncodings, Symmetries>::size(int) {
  return Tensor::Dimensions::total_size;
}

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
void ValueNetworkHead<TensorEncodings, Symmetries>::uniform_init(float* data, int) {
  auto dst = detail::make_tensor_map<Tensor>(data);
  dst.setConstant(1.0f / eigen_util::size(dst));
}

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
template <typename InitParams>
void ActionValueNetworkHead<TensorEncodings, Symmetries>::load(float* data, Tensor& src,
                                                              const InitParams& params) {
  group::element_t inv_sym = Game::SymmetryGroup::inverse(params.sym);
  eigen_util::right_rotate(src, params.active_seat);
  Symmetries::apply(src, inv_sym, params.frame);

  auto dst = detail::make_per_action_tensor_map<Tensor>(data, params.valid_moves.size());
  int i = 0;
  for (Move move : params.valid_moves) {
    auto index = PolicyEncoding::to_index(params.frame, move);
    dst.chip(i++, 0) = eigen_util::chip_recursive(src, index);
  }

  eigen_util::rowwise_softmax_in_place(dst);
}

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
int ActionValueNetworkHead<TensorEncodings, Symmetries>::size(int num_valid_moves) {
  using Shape = Tensor::Dimensions;
  return (Tensor::Dimensions::total_size / eigen_util::extract_dim_v<0, Shape>)*num_valid_moves;
}

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
void ActionValueNetworkHead<TensorEncodings, Symmetries>::uniform_init(float* data, int num_valid_moves) {
  auto dst = detail::make_per_action_tensor_map<Tensor>(data, num_valid_moves);
  dst.setConstant(1.0f / TensorEncodings::Game::Constants::kNumPlayers);
}

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
template <typename InitParams>
void ValueUncertaintyNetworkHead<TensorEncodings, Symmetries>::load(float* data, Tensor& src,
                                                                   const InitParams& params) {
  eigen_util::right_rotate(src, params.active_seat);
  auto dst = detail::make_tensor_map<Tensor>(data);
  dst = src;
}

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
int ValueUncertaintyNetworkHead<TensorEncodings, Symmetries>::size(int) {
  return Tensor::Dimensions::total_size;
}

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
void ValueUncertaintyNetworkHead<TensorEncodings, Symmetries>::uniform_init(float* data, int) {
  auto dst = detail::make_tensor_map<Tensor>(data);
  dst.setConstant(0.5f);
}

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
template <typename InitParams>
void ActionValueUncertaintyNetworkHead<TensorEncodings, Symmetries>::load(float* data, Tensor& src,
                                                                         const InitParams& params) {
  group::element_t inv_sym = Game::SymmetryGroup::inverse(params.sym);
  eigen_util::right_rotate(src, params.active_seat);
  Symmetries::apply(src, inv_sym, params.frame);

  auto dst = detail::make_per_action_tensor_map<Tensor>(data, params.valid_moves.size());
  int i = 0;
  for (Move move : params.valid_moves) {
    auto index = PolicyEncoding::to_index(params.frame, move);
    dst.chip(i++, 0) = eigen_util::chip_recursive(src, index);
  }
}

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
int ActionValueUncertaintyNetworkHead<TensorEncodings, Symmetries>::size(int num_valid_moves) {
  using Shape = Tensor::Dimensions;
  return (Tensor::Dimensions::total_size / eigen_util::extract_dim_v<0, Shape>)*num_valid_moves;
}

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
void ActionValueUncertaintyNetworkHead<TensorEncodings, Symmetries>::uniform_init(float* data,
                                                                                 int num_valid_moves) {
  auto dst = detail::make_per_action_tensor_map<Tensor>(data, num_valid_moves);
  dst.setConstant(0.5f);
}

}  // namespace core
