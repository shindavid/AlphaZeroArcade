#pragma once

#include <bitset>
#include <concepts>
#include <cstdint>
#include <type_traits>

#include <core/BasicTypes.hpp>
#include <core/GameStateHistory.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>
#include <util/MetaProgramming.hpp>

namespace core {

namespace concepts {

template <class T, class GameState>
concept AuxTarget = requires(const typename GameState::Data& cur_state,
                             const typename GameState::Data& final_state,
                             typename T::Tensor& tensor)
{
  { util::decay_copy(T::kName) } -> std::same_as<const char*>;
  requires eigen_util::FixedTensorConcept<typename T::Tensor>;
  { T::tensorize(tensor, cur_state, final_state) } -> std::same_as<void>;
};

template <class GameState, typename T>
struct IsAuxTargetList {
  static constexpr bool value = false;
};

template <typename GameState, AuxTarget<GameState>... Ts>
struct IsAuxTargetList<GameState, mp::TypeList<Ts...>> {
  static constexpr bool value = true;
};

template <typename T, class GameState>
concept AuxTargetList = IsAuxTargetList<GameState, T>::value;

}  // namespace concepts

/*
 * All Tensorizor classes must satisfy the TensorizorConcept concept.
 *
 * A Tensorizor is responsible for converting a GameState into a Tensor.
 *
 * AlphaGo includes a history of the past 7 game states in the input tensor. If you want to include
 * history like this, the Tensorizor class is the appropriate place to maintain that state.
 */
template <class Tensorizor, class GameState>
concept TensorizorConcept = requires(Tensorizor tensorizor,
                                     typename Tensorizor::InputTensor& input,
                                     const typename GameState::Data& game_state_data,
                                     const typename Tensorizor::GameStateHistory& history)
{
  requires eigen_util::FixedTensorConcept<typename Tensorizor::InputTensor>;
  requires core::concepts::AuxTargetList<typename Tensorizor::AuxTargetList, GameState>;
  requires core::concepts::GameStateHistory<typename Tensorizor::GameStateHistory>;

  /*
    * Takes an eigen Tensor reference and writes to it.
    */
  { Tensorizor::tensorize(input, game_state_data, history) };
};

}  // namespace core
