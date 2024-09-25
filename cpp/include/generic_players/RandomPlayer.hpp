#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <util/EigenUtil.hpp>
#include <util/Random.hpp>

namespace generic {

/*
 * RandomPlayer always chooses uniformly at random among the set of legal moves.
 */
template<core::concepts::Game Game>
class RandomPlayer : public core::AbstractPlayer<Game> {
public:
  using base_t = core::AbstractPlayer<Game>;
  using State = Game::State;
  using ActionMask = Game::Types::ActionMask;

  core::ActionResponse get_action_response(const State&, const ActionMask& mask) override {
    return bitset_util::choose_random_on_index(mask);
  }
};

}  // namespace generic
