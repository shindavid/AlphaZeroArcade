#pragma once

#include <array>
#include <string>

#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/BasicTypes.hpp>

namespace common {

template<GameStateConcept GameState>
class AbstractPlayer {
public:
  using GameStateTypes = common::GameStateTypes<GameState>;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using ActionMask = typename GameStateTypes::ActionMask;
  using player_array_t = std::array<AbstractPlayer*, GameState::kNumPlayers>;
  using player_name_array_t = typename GameStateTypes::player_name_array_t;

  AbstractPlayer(const std::string &name) : name_(name) {}
  virtual ~AbstractPlayer() = default;
  void set_name(const std::string &name) { name_ = name; }
  std::string get_name() const { return name_; }
  const player_name_array_t& get_player_names() const { return player_names_; }
  game_id_t get_game_id() const { return game_id_; }
  player_index_t get_my_seat() const { return my_seat_; }

  void init_game(game_id_t game_id, const player_name_array_t& player_names, player_index_t seat_assignment) {
    game_id_ = game_id;
    player_names_ = player_names;
    my_seat_ = seat_assignment;
  }

  virtual void start_game() {}
  virtual void receive_state_change(player_index_t, const GameState&, action_index_t, const GameOutcome&) {}
  virtual action_index_t get_action(const GameState&, const ActionMask&) = 0;

private:
  std::string name_;
  player_name_array_t player_names_;
  game_id_t game_id_ = -1;
  player_index_t my_seat_ = -1;
};

}  // namespace common
