#pragma once

#include <Eigen/Core>

namespace common {

/*
 * Represents the result of a game, as a length-t array of non-negative floats, where t is the number of players in the
 * game.
 *
 * If the result represents a terminal game state, the array will have sum 1. Normally, one slot in the array,
 * corresponding to the winner, will equal 1, and the other slots will equal 0. In the even of a draw, the tied
 * players will typically each equal the same fractional value.
 *
 * If the game is not yet over, the result will have all zeros.
 *
 * I chose to build on top of Eigen::Matrix, to take advantage of the built-in sum() and operator+ methods. It might
 * be better to just use std::array instead, to avoid unnecessary header dependencies.
 */
template <int tNumPlayers>
class GameResult : public Eigen::Matrix<float, tNumPlayers, 1> {
public:
  bool is_terminal() const { return this->sum() > 0; }
};

template<typename T> struct is_game_result { static const bool value = false; };
template<int N> struct is_game_result<GameResult<N>> { static const bool value = true; };
template<typename T> inline constexpr bool is_game_result_v = is_game_result<T>::value;
template <typename T> concept is_game_result_c = is_game_result_v<T>;

}  // namespace common
