#pragma once

#include "core/concepts/TensorEncodingsConcept.hpp"
#include "util/MetaProgramming.hpp"

// Contains definitions for network heads.
//
// Each *NetworkHead class corresponds to an output head of the neural network. Each class must
// satisfy the core::concepts::NetworkHead concept.
namespace core {

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
struct PolicyNetworkHead {
  static constexpr char kName[] = "policy";
  using Game = TensorEncodings::Game;
  using MoveSet = Game::MoveSet;
  using Move = Game::Move;
  using PolicyEncoding = TensorEncodings::PolicyEncoding;
  using Tensor = PolicyEncoding::Tensor;

  template <typename InitParams>
  static void load(float* data, Tensor& src, const InitParams& params);

  static int size(int num_valid_moves);
  static void uniform_init(float* data, int num_valid_moves);
};

// NOTE: If we add a game where we produce non-logit value predictions, we should modify this to
// allow customization. We will likely need this in single-player games.
template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
struct ValueNetworkHead {
  static constexpr char kName[] = "value";
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using Tensor = GameResultEncoding::Tensor;

  template <typename InitParams>
  static void load(float* data, Tensor& src, const InitParams& params);

  static int size(int num_valid_moves);
  static void uniform_init(float* data, int num_valid_moves);
};

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
struct LcZeroValueNetworkHead {
  static constexpr char kName[] = "value";
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using Tensor = GameResultEncoding::Tensor;

  template <typename InitParams>
  static void load(float* data, Tensor& src, const InitParams& params);

  static int size(int num_valid_moves);
  static void uniform_init(float* data, int num_valid_moves);
};

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
struct ActionValueNetworkHead {
  static constexpr char kName[] = "action_value";
  using Game = TensorEncodings::Game;
  using MoveSet = Game::MoveSet;
  using Move = Game::Move;
  using PolicyEncoding = TensorEncodings::PolicyEncoding;
  using Tensor = TensorEncodings::ActionValueEncoding::Tensor;

  template <typename InitParams>
  static void load(float* data, Tensor& src, const InitParams& params);

  static int size(int num_valid_moves);
  static void uniform_init(float* data, int num_valid_moves);
};

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
struct ValueUncertaintyNetworkHead {
  static constexpr char kName[] = "uncertainty";
  using Tensor = TensorEncodings::WinShareTensor;

  template <typename InitParams>
  static void load(float* data, Tensor& src, const InitParams& params);

  static int size(int num_valid_moves);
  static void uniform_init(float* data, int num_valid_moves);
};

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
struct ActionValueUncertaintyNetworkHead {
  static constexpr char kName[] = "action_value_uncertainty";
  using Game = TensorEncodings::Game;
  using MoveSet = Game::MoveSet;
  using Move = Game::Move;
  using PolicyEncoding = TensorEncodings::PolicyEncoding;
  using Tensor = TensorEncodings::ActionValueEncoding::Tensor;

  template <typename InitParams>
  static void load(float* data, Tensor& src, const InitParams& params);

  static int size(int num_valid_moves);
  static void uniform_init(float* data, int num_valid_moves);
};

// Per-action child embedding e_i = ChildEmbeddingHead([0, 0, 0, P_i, AV_i, AU_i; z_a_i]),
// computed on the GPU. Shape (kPolicySize, kEmbedDim). Loaded into Edge::e_cached for each
// valid move; the parent's BackupNN accumulator is then formed CPU-side as the sum over valid
// moves. Subsequent NNUE-style subtract-add updates use the CPU-side
// BackupNNEvaluator::compute_child_embedding(), which shares weights with this head (the same
// trained tensor is exported both into the main TensorRT graph and as orphan nnue/* initializers).
template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries,
          typename BackupNetDims>
struct ChildEmbeddingNetworkHead {
  static constexpr char kName[] = "child_embedding";
  using Game = TensorEncodings::Game;
  using MoveSet = Game::MoveSet;
  using Move = Game::Move;
  using PolicyEncoding = TensorEncodings::PolicyEncoding;
  static constexpr int kEmbedDim = BackupNetDims::kEmbedDim;
  using Shape = eigen_util::extend_shape_t<typename PolicyEncoding::Shape, kEmbedDim>;
  using Tensor = eigen_util::FTensor<Shape>;

  template <typename InitParams>
  static void load(float* data, Tensor& src, const InitParams& params);

  static int size(int num_valid_moves);
  static void uniform_init(float* data, int num_valid_moves);
};

namespace alpha0 {

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
struct StandardNetworkHeads {
  using Game = TensorEncodings::Game;
  using PolicyEncoding = TensorEncodings::PolicyEncoding;
  using PolicyHead = PolicyNetworkHead<TensorEncodings, Symmetries>;
  using ValueHead = ValueNetworkHead<TensorEncodings, Symmetries>;
  using ActionValueHead = ActionValueNetworkHead<TensorEncodings, Symmetries>;

  using List = mp::TypeList<PolicyHead, ValueHead, ActionValueHead>;
};

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries>
using StandardNetworkHeadsList = StandardNetworkHeads<TensorEncodings, Symmetries>::List;

}  // namespace alpha0

namespace beta0 {

template <core::concepts::TensorEncodings TensorEncodings, typename Symmetries,
          typename BackupNetDims>
struct StandardNetworkHeads {
  using Game = TensorEncodings::Game;
  using PolicyEncoding = TensorEncodings::PolicyEncoding;
  using PolicyHead = PolicyNetworkHead<TensorEncodings, Symmetries>;
  using ValueHead = ValueNetworkHead<TensorEncodings, Symmetries>;
  using UncertaintyHead = ValueUncertaintyNetworkHead<TensorEncodings, Symmetries>;
  using ActionValueHead = ActionValueNetworkHead<TensorEncodings, Symmetries>;
  using ActionValueUncertaintyHead = ActionValueUncertaintyNetworkHead<TensorEncodings, Symmetries>;
  using ChildEmbeddingHead =
    ChildEmbeddingNetworkHead<TensorEncodings, Symmetries, BackupNetDims>;

  // TODO: add z_a (ActionLatentHead) and z_s (StaticLatentHead) heads, for per-action and static
  // latent variables for the BackupNet. Until then, the BackupNet's z_s and z_a inputs are
  // zero-filled at search time, which means the network's per-state predictions are
  // calibration-meaningless (only its overall scale / bias survives). The integration tests
  // exercise the wiring, not the calibration.

  using List = mp::TypeList<PolicyHead, ValueHead, UncertaintyHead, ActionValueHead,
                            ActionValueUncertaintyHead, ChildEmbeddingHead>;
};

}  // namespace beta0

}  // namespace core

#include "inline/core/NetworkHeads.inl"
