#pragma once

#include <bitset>
#include <tuple>
#include <utility>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <common/BasicTypes.hpp>
#include <util/BitSet.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenTorch.hpp>
#include <util/EigenUtil.hpp>
#include <util/Math.hpp>
#include <util/TorchUtil.hpp>

namespace common {

/*
 * Represents the outcome of a game, as a length-t array of non-negative floats, where t is the number of players in
 * the game.
 *
 * If the outcome represents a terminal game state, the array will have sum 1. Normally, one slot in the array,
 * corresponding to the winner, will equal 1, and the other slots will equal 0. In the even of a draw, the tied
 * players will typically each equal the same fractional value.
 *
 * If the game is not yet over, the outcome will have all zeros.
 */
template<int NumPlayers> using GameOutcome_ = Eigen::Array<torch_util::dtype, NumPlayers, 1>;
template<int NumPlayers> bool is_terminal_outcome(const GameOutcome_<NumPlayers>& outcome) { return outcome.sum() > 0; }
template<int NumPlayers> auto make_non_terminal_outcome() { GameOutcome_<NumPlayers> o; o.setZero(); return o; }

template<typename GameState>
struct GameStateTypes {
  using dtype = torch_util::dtype;

  using PolicyShape = typename GameState::PolicyShape;
  using ValueShape = eigen_util::Shape<GameState::kNumPlayers>;

  static constexpr int kNumPlayers = GameState::kNumPlayers;
  static constexpr int kNumGlobalActions = PolicyShape::total_size;
  static constexpr int kMaxNumLocalActions = GameState::kMaxNumLocalActions;

  using GameOutcome = GameOutcome_<kNumPlayers>;

  template <int NumRows> using PolicyShapeN = eigen_util::prepend_dim_t<NumRows, PolicyShape>;
  template <int NumRows> using PolicyTensorN = eigentorch::TensorFixedSize<dtype, PolicyShapeN<NumRows>>;
  using PolicyTensor = eigentorch::TensorFixedSize<dtype, PolicyShape>;
  using PolicyEigenTensor = typename PolicyTensor::EigenType;

  template <int NumRows> using ValueShapeN = eigen_util::prepend_dim_t<NumRows, ValueShape>;
  template <int NumRows> using ValueTensorN = eigentorch::TensorFixedSize<dtype, ValueShapeN<NumRows>>;
  using ValueTensor = eigentorch::TensorFixedSize<dtype, ValueShape>;
  using ValueEigenTensor = typename ValueTensor::EigenType;

  // flattened versions of PolicyTensor and ValueTensor
  using PolicyArray = Eigen::Array<dtype, kNumGlobalActions, 1>;
  using ValueArray = Eigen::Array<dtype, kNumPlayers, 1>;

  using ValueProbDistr = Eigen::Array<dtype, kNumPlayers, 1>;
  using LocalPolicyLogitDistr = Eigen::Array<dtype, Eigen::Dynamic, 1, 0, kMaxNumLocalActions>;
  using LocalPolicyProbDistr = Eigen::Array<dtype, Eigen::Dynamic, 1, 0, kMaxNumLocalActions>;

  using DynamicPolicyTensor = eigentorch::Tensor<dtype, PolicyShape::count + 1>;
  using DynamicValueTensor = eigentorch::Tensor<dtype, ValueShape::count + 1>;

  using ActionMask = std::bitset<kNumGlobalActions>;
  using player_name_array_t = std::array<std::string, kNumPlayers>;

  static LocalPolicyProbDistr global_to_local(const PolicyEigenTensor& policy, const ActionMask& mask);
  static void global_to_local(const PolicyEigenTensor& policy, const ActionMask& mask, LocalPolicyProbDistr& out);

  static PolicyEigenTensor local_to_global(const LocalPolicyProbDistr& policy, const ActionMask& mask);
  static void local_to_global(const LocalPolicyProbDistr& policy, const ActionMask& mask, PolicyEigenTensor& out);

  /*
   * Provides variable bindings, so that we can specify certain config variables as expressions of game parameters.
   * See util/Math.hpp
   *
   * Bindings:
   *
   * "b" -> kMaxNumLocalActions (max _b_ranching factor)
   */
  static math::var_bindings_map_t get_var_bindings();
};

template<typename Tensorizor>
struct TensorizorTypes {
  using dtype = torch_util::dtype;
  static constexpr int kMaxNumSymmetries = Tensorizor::kMaxNumSymmetries;
  using InputShape = typename Tensorizor::InputShape;

  using SymmetryIndexSet = std::bitset<kMaxNumSymmetries>;

  template <int NumRows> using InputShapeN = eigen_util::prepend_dim_t<NumRows, InputShape>;
  template <int NumRows> using InputTensorN = eigentorch::TensorFixedSize<dtype, InputShapeN<NumRows>>;
  using InputTensor = eigentorch::TensorFixedSize<dtype, InputShape>;
  using DynamicInputTensor = eigentorch::Tensor<dtype, InputShape::count + 1>;
};

template<typename GameState>
struct StateEvaluationKey {
  GameState state;
  symmetry_index_t sym_index;

  bool operator==(const StateEvaluationKey& other) const {
    return state == other.state && sym_index == other.sym_index;
  }
};

}  // namespace common

template <typename GameState>
struct std::hash<common::StateEvaluationKey<GameState>> {
  std::size_t operator()(const common::StateEvaluationKey<GameState> ssi) const {
    return util::tuple_hash(std::make_tuple(ssi.state, ssi.sym_index));
  }
};

#include <common/inl/DerivedTypes.inl>
