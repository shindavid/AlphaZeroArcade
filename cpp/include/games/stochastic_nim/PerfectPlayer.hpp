#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <games/stochastic_nim/Constants.hpp>
#include <games/stochastic_nim/Game.hpp>
#include <util/EigenUtil.hpp>

namespace stochastic_nim {

class PerfectPlayer : public core::AbstractPlayer<stochastic_nim::Game> {
 public:

  using base_t = core::AbstractPlayer<stochastic_nim::Game>;
  using Constants = stochastic_nim::Game::Constants;
  using Rules = stochastic_nim::Game::Rules;
  using Types = stochastic_nim::Game::Types;
  using ActionMask = Types::ActionMask;
  using State = stochastic_nim::Game::State;
  using ChanceDistribution = Types::ChanceDistribution;

  static constexpr size_t kNumModes = Constants::kNumActionsPerMode::size();
  using StateActionShape =
      eigen_util::Shape<stochastic_nim::kStartingStones + 1, Constants::kNumPlayers, kNumModes,
                        Types::kMaxNumActions>;
  using StateActionTensor = eigen_util::FTensor<StateActionShape>;

  struct Params {
    /*
     * The strength parameter controls how well the player plays. It is either 0 (random) or
     * 1 (perfect).
     */
    int strength = 1;
    bool verbose = false;
    auto make_options_description(){};
  };

  PerfectPlayer(const Params&);
  void start_game() override {};
  void receive_state_change(core::seat_index_t, const State&, core::action_t) override {};
  ActionResponse get_action_response(const State&, const ActionMask&) override;
  StateActionTensor get_state_action_tensor() const { return state_action_tensor_; }

  private:
    StateActionTensor state_action_tensor_;
    void update_state_action_tensor();
    void update_player_state_action_tensor(int stones_left, core::seat_index_t);
    void update_chance_state_action_tensor(int stones_left, core::seat_index_t);
};

} // namespace stochastic_nim

#include <inline/games/stochastic_nim/PerfectPlayer.inl>