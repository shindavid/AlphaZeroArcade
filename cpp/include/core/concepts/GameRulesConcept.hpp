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

  // analyze() returns a RulesResult. The RulesResult either tells us the set of legal moves from
  // that state (if the game is not terminal), or the outcome of the game (if the game is
  // terminal).
  //
  // last_move_info.action is the last action that was taken (whether by a player or a
  // chance-event), and last_move_info.player is the seat that was active when that action was
  // taken. For player events, last_move_info.player will be the seat of the player who took the
  // action. For chance events, last_move_info.player will be the seat of the player who was
  // active before the chance event.
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
