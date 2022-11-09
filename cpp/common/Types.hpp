#pragma once

#include <Eigen/Core>

#include <array>
#include <cstdint>
#include <string>

#include <util/BitSet.hpp>

namespace common {

using player_index_t = int8_t;
using action_index_t = int32_t;

template <int tNumPlayers>
class GameResult : public Eigen::Matrix<float, tNumPlayers, 1> {
public:
  bool is_terminal() const { return this->sum() > 0; }
};

template<typename T> struct is_game_result { static const bool value = false; };
template<int N> struct is_game_result<GameResult<N>> { static const bool value = true; };
template<typename T> inline constexpr bool is_game_result_v = is_game_result<T>::value;
template <typename T> concept is_game_result_c = is_game_result_v<T>;

/*
 * All GameState classes must satisfy the GameStateConcept concept.
 */
template <class S>
concept GameStateConcept = requires(S state) {
  /*
   * The number of players in the game.
   */
  { S::get_num_players() } -> std::same_as<int>;

  /*
   * Return the total number of global actions in the game.
   *
   * For go, this is 19*19+1 = 362 (+1 because you can pass).
   * For connect-four, this is 7.
   */
  { S::get_num_global_actions() } -> std::same_as<int>;

  /*
   * Return the current player.
   */
  { state.get_current_player() } -> std::same_as<player_index_t>;

  /*
   * Apply a given action to the state, and return a GameResult.
   */
  { state.apply_move(action_index_t()) } -> is_game_result_c;

  /*
   * Get the valid actions, as a util::BitSet.
   */
  { state.get_valid_actions() } -> is_bit_set_c;

  /*
   * A compact string representation, used for debugging purposes in conjunction with javascript visualizer.
   */
  { state.compact_repr() } -> std::same_as<std::string>;

  /*
   * Must be hashable.
   */
  { std::hash<S>{}(state) } -> std::convertible_to<std::size_t>;
};

template<GameStateConcept GameState>
class AbstractPlayer {
public:
  using Result = GameResult<GameState::get_num_players()>;
  using ActionMask = util::BitSet<GameState::get_num_global_actions()>;
  using player_array_t = std::array<AbstractPlayer*, GameState::get_num_players()>;

  AbstractPlayer(const std::string& name) : name_(name) {}
  virtual ~AbstractPlayer() = default;
  void set_name(const std::string& name) { name_ = name; }
  std::string get_name() const { return name_; }

  virtual void start_game(const player_array_t& players, player_index_t seat_assignment) {}
  virtual void receive_state_change(player_index_t, const GameState&, action_index_t, const Result&) {}
  virtual action_index_t get_action(const GameState&, const ActionMask&) = 0;

private:
  std::string name_;
};

}  // namespace common
