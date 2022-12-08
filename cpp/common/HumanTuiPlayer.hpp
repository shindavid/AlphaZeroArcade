#pragma once

#include <ostream>

#include <common/AbstractPlayer.hpp>
#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>

namespace common {

template<GameStateConcept GameState_>
class HumanTuiPlayer : public AbstractPlayer<GameState_> {
public:
  using base_t = AbstractPlayer<GameState_>;
  using GameState = GameState_;
  using GameStateTypes = GameStateTypes_<GameState>;

  using ActionMask = typename GameStateTypes::ActionMask;
  using GameResult = typename GameStateTypes::GameResult;
  using player_name_array_t = typename GameStateTypes::player_name_array_t;
  using player_array_t = typename base_t::player_array_t;

  HumanTuiPlayer() : base_t("Human") {}
  void start_game(const player_array_t& players, common::player_index_t seat_assignment) override;
  void receive_state_change(common::player_index_t, const GameState&, common::action_index_t, const GameResult&) override;
  common::action_index_t get_action(const GameState&, const ActionMask&) override;

private:
  void xprintf_switch(const GameState&);
  void print_state(const GameState&);

  std::ostringstream buf_;
  player_name_array_t player_names_;
  common::player_index_t my_index_ = -1;
  common::action_index_t last_action_ = -1;
};

}  // namespace common

#include <common/inl/HumanTuiPlayer.inl>
