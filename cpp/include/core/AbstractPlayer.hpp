#pragma once

#include "core/BasicTypes.hpp"
#include "core/Constants.hpp"
#include "core/concepts/GameConcept.hpp"

#include <array>
#include <string>

namespace core {

/*
 * Base class for all players.
 *
 * There are 4 main virtual functions to override:
 *
 * - start_game()
 * - receive_state_change()
 * - get_action_response()
 * - end_game()
 *
 * start_game() and end_game() are called when a game starts or ends. A single player might play
 * multiple games in succession, so you should override this method if there is state that you want
 * to clear between games.
 *
 * receive_state_change() is called when the game state changes. This is where you should update
 * your internal state. Note that you get this callback even after you make your own turn as a sort
 * of "echo" of your own action.
 *
 * get_action_response() is called when it is your turn to make a move. This is where you should
 * return the action that you want to take.
 *
 * TODO: for imperfect-information games, these methods should accept an "information set", rather
 * than a State. Flush out the details of this if/when we get there.
 */
template <concepts::Game Game>
class AbstractPlayer {
 public:
  using State = Game::State;
  using GameResultTensor = Game::Types::GameResultTensor;
  using ActionMask = Game::Types::ActionMask;
  using ActionRequest = Game::Types::ActionRequest;
  using ActionResponse = Game::Types::ActionResponse;
  using ChanceEventHandleRequest = Game::Types::ChanceEventHandleRequest;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using player_array_t = std::array<AbstractPlayer*, Game::Constants::kNumPlayers>;
  using player_name_array_t = Game::Types::player_name_array_t;

  virtual ~AbstractPlayer() = default;
  void set_name(const std::string& name) { name_ = name; }
  const std::string& get_name() const { return name_; }
  const player_name_array_t& get_player_names() const { return player_names_; }
  game_id_t get_game_id() const { return game_id_; }
  seat_index_t get_my_seat() const { return my_seat_; }

  void init_game(game_id_t game_id, const player_name_array_t& player_names,
                 seat_index_t seat_assignment);

  // start_game() should return false if the player refuses to play the game.
  virtual bool start_game() { return true; }

  virtual void receive_state_change(seat_index_t, const State&, action_t) {}

  /*
   * In games with chance events, this method is called before the chance event occurs. This gives
   * the player a chance to output action value targets to be used for training.
   */
  virtual core::yield_instruction_t handle_chance_event(const ChanceEventHandleRequest&) {
    return core::kContinue;
  }

  /*
   * request.state is guaranteed to be identical to the State last received via
   * receive_state_change().
   */
  virtual ActionResponse get_action_response(const ActionRequest& request) = 0;

  /*
   * The State passed in here is guaranteed to be identical to the State last received via
   * receive_state_change().
   */
  virtual void end_game(const State&, const GameResultTensor&) {}

  // Override this to return true if you don't want GameServer to display a progress bar.
  virtual bool disable_progress_bar() const { return false; }

 private:
  std::string name_;
  player_name_array_t player_names_;
  game_id_t game_id_ = -1;
  seat_index_t my_seat_ = -1;
};

}  // namespace core

#include "inline/core/AbstractPlayer.inl"
