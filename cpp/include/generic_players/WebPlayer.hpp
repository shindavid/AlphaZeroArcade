#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"
#include "generic_players/WebPlayerBase.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <boost/asio.hpp>
#include <boost/json.hpp>
#include <boost/process.hpp>

namespace generic {

template <core::concepts::Game Game>
class WebPlayer : public WebPlayerBase<Game> {
 public:
  using GameClass = Game;
  using GameTypes = Game::Types;
  using base_t = core::AbstractPlayer<Game>;
  using IO = Game::IO;
  using State = Game::State;
  using ActionMask = Game::Types::ActionMask;
  using ActionRequest = Game::Types::ActionRequest;
  using ActionResponse = Game::Types::ActionResponse;
  using GameResultTensor = Game::Types::GameResultTensor;

  static_assert(core::concepts::WebGameIO<IO, GameTypes>, "IO must satisfy WebGameIO");

  bool start_game() override;
  void receive_state_change(core::seat_index_t, const State&, core::action_t) override;
  ActionResponse get_action_response(const ActionRequest&) override;
  void end_game(const State&, const GameResultTensor&) override;
  bool disable_progress_bar() const override { return true; }
};

}  // namespace generic

#include "inline/generic_players/WebPlayer.inl"
