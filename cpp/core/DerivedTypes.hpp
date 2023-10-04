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
#include <util/MetaProgramming.hpp>
#include <util/TorchUtil.hpp>

namespace core {

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

  static constexpr int kNumPlayers = GameState::kNumPlayers;
  static constexpr int kNumGlobalActionsBound = GameState::ActionShape::total_size;
  static constexpr int kMaxNumLocalActions = GameState::kMaxNumLocalActions;
  static constexpr int kMaxNumSymmetries = GameState::kMaxNumSymmetries;

  using SymmetryIndexSet = std::bitset<kMaxNumSymmetries>;

  using GameOutcome = core::GameOutcome<kNumPlayers>;

  using PolicyShape = typename GameState::ActionShape;
  using ValueShape = eigen_util::Shape<kNumPlayers>;

  using PolicyTensor = Eigen::TensorFixedSize<dtype, PolicyShape, Eigen::RowMajor>;
  using ValueTensor = Eigen::TensorFixedSize<dtype, ValueShape, Eigen::RowMajor>;

  using PolicyArray = Eigen::Array<dtype, kNumGlobalActionsBound, 1>;
  using ValueArray = Eigen::Array<dtype, kNumPlayers, 1>;
  using LocalPolicyArray = Eigen::Array<dtype, Eigen::Dynamic, 1, 0, kMaxNumLocalActions>;

  using DynamicPolicyTensor = Eigen::Tensor<dtype, PolicyShape::count + 1, Eigen::RowMajor>;
  using DynamicValueTensor = Eigen::Tensor<dtype, ValueShape::count + 1, Eigen::RowMajor>;

  using Action = std::array<int64_t, size_t(PolicyShape::count)>;
  using ActionMask = Eigen::TensorFixedSize<bool, PolicyShape, Eigen::RowMajor>;
  using player_name_array_t = std::array<std::string, kNumPlayers>;

  static LocalPolicyArray global_to_local(const PolicyTensor& policy, const ActionMask& mask);
  static void global_to_local(const PolicyTensor& policy, const ActionMask& mask, LocalPolicyArray& out);

  static PolicyTensor local_to_global(const LocalPolicyArray& policy, const ActionMask& mask);
  static void local_to_global(const LocalPolicyArray& policy, const ActionMask& mask, PolicyTensor& out);

  static Action get_nth_valid_action(const ActionMask& valid_actions, int n);

  static void nullify_action(Action& action);
  static bool is_nullified(const Action& action);

  /*
   * is_valid_action() validates that 0 <= action[i] <= PolicyShape::dimension(i) for all i.
   *
   * If valid_actions is passed in, then validates further that valid_actions[action] is true.
   *
   * The validate_action() functions are similar, but throw an exception if the action is invalid.
   */
  static bool is_valid_action(const Action& action);
  static bool is_valid_action(const Action& action, const ActionMask& valid_actions);
  static void validate_action(const Action& action);
  static void validate_action(const Action& action, const ActionMask& valid_actions);

  /*
   * Rescales policy to sum to 1.
   *
   * If policy is all zeros, then policy is altered to be uniformly positive where mask is true.
   */
  static void normalize(const ActionMask& mask, PolicyTensor& policy);

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

template<typename T>
struct ExtractAuxTargetTensor {
  using type = typename T::Tensor;
};

template<typename Tensorizor>
struct TensorizorTypes {
  using InputTensor = typename Tensorizor::InputTensor;
  using InputShape = eigen_util::extract_shape_t<InputTensor>;
  using InputScalar = typename InputTensor::Scalar;
  using AuxTargetList = typename Tensorizor::AuxTargetList;

  using AuxTargetTensorList = mp::TransformTypeList_t<ExtractAuxTargetTensor, AuxTargetList>;
  using AuxTargetTensorTuple = mp::TypeListToTuple_t<AuxTargetTensorList>;
};

}  // namespace core

#include <core/inl/DerivedTypes.inl>
