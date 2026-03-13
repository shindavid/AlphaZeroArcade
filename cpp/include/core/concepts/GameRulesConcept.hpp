#pragma once

#include "core/BasicTypes.hpp"
#include "core/RulesResult.hpp"

#include <concepts>

namespace core {
namespace concepts {

template <typename GR, typename GameTypes, typename State>
concept GameRules = requires(const State& const_state, const State& prev_state, State& state,
                             const core::MoveInfo& last_move_info, action_mode_t action_mode) {
  { GR::init_state(state) };
  { GR::get_action_mode(const_state) } -> std::same_as<core::action_mode_t>;

  // Assumes the state is in player mode.
  { GR::get_current_player(const_state) } -> std::same_as<core::seat_index_t>;
  { GR::apply(state, core::action_t{}) };

  // TODO: make this function constexpr
  { GR::is_chance_mode(action_mode) } -> std::same_as<bool>;

  // Assumes the state is in chance mode.
  {
    GR::get_chance_distribution(const_state)
  } -> std::same_as<typename GameTypes::ChanceDistribution>;

  // Analyzes the current state and recent move info to determine its terminal status and valid actions.
  //
  // Returns a RulesResult containing either the outcome of the game (if terminal)
  // or the set of legal moves (if non-terminal).
  //
  // 'last_move_info' parameters:
  // - action: the last action that was taken whether by a player or a chance-event
  // - For non-chance events: 'player' is the seat of the player who took the action.
  // - For chance events: 'player' is the seat of the player who was active before the chance
  //   event occurred.
  // - If the previous move is unknown, pass a default MoveInfo (e.g., {-1, -1}).
  { GR::analyze(const_state, last_move_info) } -> std::same_as<core::RulesResult<GameTypes>>;
  { GR::analyze(const_state, last_move_info) } -> std::same_as<core::RulesResult<GameTypes>>;

  // Most classes can simply implement this as a call to the copy assignment operator. Others may
  // want to take advantage of the fact that other_states is a previous state in the same game.
  //
  // For example, for chess, the state stores a history of all past boards to support the
  // repetition rule. Resetting to a prior state can be implemented by simply truncating the
  // history, which is more efficient than copying the entire state.
  { GR::backtrack_state(state, prev_state) };
};

}  // namespace concepts
}  // namespace core
