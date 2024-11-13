#pragma once

#include <core/concepts/Game.hpp>

/*
 * This file contains game transformations that can be applied to a game to modify its behavior.
 */

namespace game_transform {

/*
 * AddStateStorage is a game transformation that adds state storage to a game by setting
 * MctsConfiguration::kStoreStates to true. mcts::Node::kStoreStates will be set to true if this
 * transformation is applied to a game. Then, the Node will store the State in its stable_data
 * during the search. If search_log_ is created and its update function is passed to the manager,
 * by calling manager_->set_post_visit_func([&] { search_log_->update(); }), then the search log
 * will be updated with the state of the root node after each search iteration.
 */

template <core::concepts::Game Game>
struct AddStateStorage : public Game {
  struct MctsConfiguration : public Game::MctsConfiguration {
    static constexpr bool kStoreStates = true;
  };
};

}  // namespace game_transform
