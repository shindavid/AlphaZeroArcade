#pragma once

#include <array>
#include <string>

#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <util/Exception.hpp>

namespace common {

/*
 * Base class for all players.
 *
 * There are 4 main virtual functions to override:
 *
 * - start_game()
 * - receive_state_change()
 * - get_action()
 * - end_game()
 *
 * start_game() and end_game() are called when a game starts or ends. A single player might play multiple games in
 * succession, so you should override this method if there is state that you want to clear between games.
 *
 * receive_state_change() is called when the game state changes. This is where you should update your internal state.
 * Note that you get this callback even after you make your own turn as a sort of "echo" of your own action.
 *
 * get_action() is called when it is your turn to make a move. This is where you should return the action that you want
 * to take.
 */
template<GameStateConcept GameState>
class AbstractPlayer {
public:
  using GameStateTypes = common::GameStateTypes<GameState>;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using ActionMask = typename GameStateTypes::ActionMask;
  using player_array_t = std::array<AbstractPlayer*, GameState::kNumPlayers>;
  using player_name_array_t = typename GameStateTypes::player_name_array_t;

  virtual ~AbstractPlayer() = default;
  void set_name(const std::string &name) { name_ = name; }
  std::string get_name() const { return name_; }
  const player_name_array_t& get_player_names() const { return player_names_; }
  game_id_t get_game_id() const { return game_id_; }
  seat_index_t get_my_seat() const { return my_seat_; }
  void init_game(game_id_t game_id, const player_name_array_t& player_names, seat_index_t seat_assignment);

  virtual void start_game() {}
  virtual void receive_state_change(seat_index_t, const GameState&, action_index_t) {}
  virtual action_index_t get_action(const GameState&, const ActionMask&) = 0;
  virtual void end_game(const GameState&, const GameOutcome&) {}

  /*
   * Some extra virtual functions that most subclasses can ignore.
   *
   * GameServer uses max_simultaneous_games() to determine how many games it can run in parallel. The default return
   * value of 0 indicates that the player can play an unlimited number of games simultaneously. Currently, we only
   * override this default for human players, which can only play one game at a time due to interface limitations.
   *
   * GameServer will invoke set_facing_human_tui_player() if there is a player in the game that is a HumanTuiPlayer.
   * If you want to do something special when you are playing against a human TUI player, you can override this method.
   * You might want to do this because you may want to print verbose information differently in this case, in order to
   * avoid interfering with the user-interface of the human player.
   *
   * is_human_tui_player() is used by GameServer to determine whether to call set_facing_human_tui_player().
   */
  virtual int max_simultaneous_games() const { return 0; }
  virtual bool is_human_tui_player() const { return false; }
  virtual void set_facing_human_tui_player() {}

private:
  std::string name_;
  player_name_array_t player_names_;
  game_id_t game_id_ = -1;
  seat_index_t my_seat_ = -1;
};

}  // namespace common

#include <common/inl/AbstractPlayer.inl>
