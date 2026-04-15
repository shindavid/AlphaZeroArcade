#pragma once

#include "beta0/Manager.hpp"
#include "beta0/concepts/SpecConcept.hpp"
#include "core/AbstractPlayer.hpp"
#include "core/ActionRequest.hpp"
#include "core/ActionResponse.hpp"
#include "core/StateChangeUpdate.hpp"
#include "search/Constants.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <memory>

namespace beta0 {

/*
 * The beta0::Player uses BetaZero MCTS to select actions.
 *
 * Note that when 2 or more identically-configured beta0::Player's are playing in the same
 * game, they can share the same MCTS tree, as an optimization. This implementation supports this
 * optimization.
 */
template <beta0::concepts::Spec Spec_>
class Player : public core::AbstractPlayer<typename Spec_::Game> {
 public:
  using BasePlayer = Player;  // needed for beta0::PlayerGeneratorBase
  using Spec = Spec_;
  using Game = Spec::Game;

  struct Params {
    Params(search::Mode);

    auto make_options_description();
  };

  using Manager = beta0::Manager<Spec>;
  using State = Game::State;
  using ActionRequest = core::ActionRequest<Game>;
  using ActionResponse = core::ActionResponse<Game>;
  using GameOutcome = Game::Types::GameOutcome;
  using StateChangeUpdate = core::StateChangeUpdate<Game>;

  struct SharedData {
    template <typename... Ts>
    SharedData(Ts&&... args) : manager(std::forward<Ts>(args)...) {}

    Manager manager;
    int num_raw_policy_starting_moves = 0;
  };
  using SharedData_sptr = std::shared_ptr<SharedData>;

  Player(const Params&, SharedData_sptr, bool owns_shared_data);

  Manager* get_manager() const { return &shared_data_->manager; }
  bool start_game() override;
  void receive_state_change(const StateChangeUpdate&) override;
  ActionResponse get_action_response(const ActionRequest&) override;
  void end_game(const State& state, const GameOutcome& results) override;

  const Params params_;
  SharedData_sptr shared_data_;
  const bool owns_shared_data_;
};

}  // namespace beta0
