#pragma once

#include <bitset>
#include <tuple>
#include <utility>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <core/BasicTypes.hpp>
#include <util/BitSet.hpp>
#include <util/CppUtil.hpp>
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
template<int NumPlayers> using GameOutcome = Eigen::Array<torch_util::dtype, NumPlayers, 1>;
template<int NumPlayers> bool is_terminal_outcome(const GameOutcome<NumPlayers>& outcome) { return outcome.sum() > 0; }
template<int NumPlayers> auto make_non_terminal_outcome() { GameOutcome<NumPlayers> o; o.setZero(); return o; }

template<typename GameState>
struct GameStateTypes {
  using dtype = torch_util::dtype;

  using PolicyShape = typename GameState::PolicyShape;
  using ValueShape = eigen_util::Shape<GameState::kNumPlayers>;

  static constexpr int kNumPlayers = GameState::kNumPlayers;
  static constexpr int kNumGlobalActions = PolicyShape::total_size;
  static constexpr int kMaxNumLocalActions = GameState::kMaxNumLocalActions;

  using GameOutcome = common::GameOutcome<kNumPlayers>;

  using PolicyTensor = Eigen::TensorFixedSize<dtype, PolicyShape, Eigen::RowMajor>;
  using ValueTensor = Eigen::TensorFixedSize<dtype, ValueShape, Eigen::RowMajor>;

  // flattened versions of PolicyTensor and ValueTensor
  using PolicyArray = Eigen::Array<dtype, kNumGlobalActions, 1>;
  using ValueArray = Eigen::Array<dtype, kNumPlayers, 1>;

  using LocalPolicyArray = Eigen::Array<dtype, Eigen::Dynamic, 1, 0, kMaxNumLocalActions>;

  using DynamicPolicyTensor = Eigen::Tensor<dtype, PolicyShape::count + 1, Eigen::RowMajor>;
  using DynamicValueTensor = Eigen::Tensor<dtype, ValueShape::count + 1, Eigen::RowMajor>;

  using ActionMask = std::bitset<kNumGlobalActions>;
  using player_name_array_t = std::array<std::string, kNumPlayers>;

  static LocalPolicyArray global_to_local(const PolicyTensor& policy, const ActionMask& mask);
  static void global_to_local(const PolicyTensor& policy, const ActionMask& mask, LocalPolicyArray& out);

  static PolicyTensor local_to_global(const LocalPolicyArray& policy, const ActionMask& mask);
  static void local_to_global(const LocalPolicyArray& policy, const ActionMask& mask, PolicyTensor& out);

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
  static constexpr int kMaxNumSymmetries = Tensorizor::kMaxNumSymmetries;
  using InputTensor = typename Tensorizor::InputTensor;
  using InputShape = eigen_util::extract_shape_t<InputTensor>;
  using InputScalar = typename InputTensor::Scalar;
  using SymmetryIndexSet = std::bitset<kMaxNumSymmetries>;
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

#include <core/inl/DerivedTypes.inl>
