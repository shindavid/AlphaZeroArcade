#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/concepts/GameConcept.hpp"

namespace generic {

/*
 * RandomPlayer always chooses uniformly at random among the set of legal moves.
 */
template <core::concepts::Game Game>
class RandomPlayer : public core::AbstractPlayer<Game> {
 public:
  using ActionRequest = Game::Types::ActionRequest;
  using ActionResponse = Game::Types::ActionResponse;

  ActionResponse get_action_response(const ActionRequest& request) override {
    return request.valid_actions.choose_random_on_index();
  }
};

}  // namespace generic
