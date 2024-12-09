#pragma once

#include <array>
#include <string>

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <core/TrainingDataWriter.hpp>
#include <util/Exception.hpp>

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
  using GameLogWriter_sptr = core::TrainingDataWriter<Game>::GameLogWriter_sptr;
  using ValueTensor = Game::Types::ValueTensor;
  using ActionMaskVariant = Game::Types::ActionMaskVariant;
  using ActionResponse = Game::Types::ActionResponse;
  using player_array_t = std::array<AbstractPlayer*, Game::Constants::kNumPlayers>;
  using player_name_array_t = Game::Types::player_name_array_t;

  virtual ~AbstractPlayer() = default;
  void set_name(const std::string& name) { name_ = name; }
  std::string get_name() const { return name_; }
  const player_name_array_t& get_player_names() const { return player_names_; }
  game_id_t get_game_id() const { return game_id_; }
  seat_index_t get_my_seat() const { return my_seat_; }
  GameLogWriter_sptr get_game_log() const { return game_log_; }
  void init_game(game_id_t game_id, const player_name_array_t& player_names,
                 seat_index_t seat_assignment, GameLogWriter_sptr game_log);

  virtual void start_game() {}
  virtual void receive_state_change(seat_index_t, const State&, action_t) {}

  /*
   * The State passed in here is guaranteed to be identical to the State last received via
   * receive_state_change().
   */
  virtual ActionResponse get_action_response(const State&, const ActionMaskVariant&) = 0;

  /*
   * The State passed in here is guaranteed to be identical to the State last received via
   * receive_state_change().
   */
  virtual void end_game(const State&, const ValueTensor&) {}

  /*
   * Some extra virtual functions that most subclasses can ignore.
   *
   * GameServer will invoke set_facing_human_tui_player() if there is a player in the game that is a
   * HumanTuiPlayer. If you want to do something special when you are playing against a human TUI
   * player, you can override this method. You might want to do this because you may want to print
   * verbose information differently in this case, in order to avoid interfering with the
   * user-interface of the human player.
   *
   * is_human_tui_player() is used by GameServer to determine whether to call
   * set_facing_human_tui_player().
   */
  virtual bool is_human_tui_player() const { return false; }
  virtual void set_facing_human_tui_player() {}

 private:
  std::string name_;
  player_name_array_t player_names_;
  game_id_t game_id_ = -1;
  seat_index_t my_seat_ = -1;
  GameLogWriter_sptr game_log_;
};

}  // namespace core

#include <inline/core/AbstractPlayer.inl>
