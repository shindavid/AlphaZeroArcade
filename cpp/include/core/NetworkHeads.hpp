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
  static constexpr char kName[] = "value_uncertainty";
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

}  // namespace core

#include "inline/core/NetworkHeads.inl"
