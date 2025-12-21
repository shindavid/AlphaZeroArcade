#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/ActionRequest.hpp"
#include "core/concepts/GameConcept.hpp"

namespace generic {

/*
 * RandomPlayer always chooses uniformly at random among the set of legal moves.
 */
template <core::concepts::Game Game>
class RandomPlayer : public core::AbstractPlayer<Game> {
 public:
  using ActionRequest = core::ActionRequest<Game>;

  core::ActionResponse get_action_response(const ActionRequest& request) override {
    if (request.aux) {
      return request.aux - 1;
    }

    core::ActionResponse response(request.valid_actions.choose_random_on_index());
    response.set_aux(response.get_action() + 1);
    return response;
  }
};

}  // namespace generic
