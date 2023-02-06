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
template<int NumPlayers> using GameOutcome_ = Eigen::Array<float, NumPlayers, 1>;
template<int NumPlayers> bool is_terminal_outcome(const GameOutcome_<NumPlayers>& outcome) { return outcome.sum() > 0; }
template<int NumPlayers> auto make_non_terminal_outcome() { GameOutcome_<NumPlayers> o; o.setZero(); return o; }

template<typename GameState>
struct GameStateTypes {
  static constexpr int kNumPlayers = GameState::kNumPlayers;
  static constexpr int kNumGlobalActions = GameState::kNumGlobalActions;
  static constexpr int kMaxNumLocalActions = GameState::kMaxNumLocalActions;

  using GameOutcome = GameOutcome_<kNumPlayers>;

  template <int NumRows> using PolicyArray = eigentorch::Array<float, NumRows, kNumGlobalActions, Eigen::RowMajor>;
  template <int NumRows> using ValueArray = eigentorch::Array<float, NumRows, kNumPlayers, Eigen::RowMajor>;

  using PolicySlab = PolicyArray<1>;
  using ValueSlab = ValueArray<1>;

  using PolicyEigenSlab = typename PolicySlab::EigenType;
  using ValueEigenSlab = typename ValueSlab::EigenType;

  using PolicyArray1D = Eigen::Array<float, kNumGlobalActions, 1>;
  using ValueArray1D = Eigen::Array<float, kNumPlayers, 1>;

  using ValueProbDistr = Eigen::Array<float, kNumPlayers, 1>;
  using LocalPolicyLogitDistr = Eigen::Array<float, Eigen::Dynamic, 1, 0, kMaxNumLocalActions>;
  using LocalPolicyProbDistr = Eigen::Array<float, Eigen::Dynamic, 1, 0, kMaxNumLocalActions>;

  using GlobalPolicyCountDistr = Eigen::Array<int, kNumGlobalActions, 1>;
  using GlobalPolicyProbDistr = Eigen::Array<float, kNumGlobalActions, 1>;

  using ActionMask = std::bitset<kNumGlobalActions>;
  using player_name_array_t = std::array<std::string, kNumPlayers>;

  static LocalPolicyProbDistr global_to_local(const PolicyArray1D& policy, const ActionMask& mask);
  static void global_to_local(const PolicyArray1D& policy, const ActionMask& mask, LocalPolicyProbDistr& out);

  static PolicyArray1D local_to_global(const LocalPolicyProbDistr& policy, const ActionMask& mask);
  static void local_to_global(const LocalPolicyProbDistr& policy, const ActionMask& mask, PolicyArray1D& out);

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
  using BaseShape = typename Tensorizor::Shape;

  using SymmetryIndexSet = std::bitset<kMaxNumSymmetries>;

  template <int NumRows> using Shape = eigen_util::to_sizes_t<
      util::concat_int_sequence_t<util::int_sequence<NumRows>, BaseShape>>;
  template <int NumRows> using InputTensor = eigentorch::TensorFixedSize<float, Shape<NumRows>>;

  using InputSlab = InputTensor<1>;
  using InputEigenSlab = typename InputSlab::EigenType;

  using DynamicInputTensor = eigentorch::Tensor<float, BaseShape::size() + 1>;
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
