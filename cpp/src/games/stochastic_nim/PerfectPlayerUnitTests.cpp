#include <games/stochastic_nim/PerfectPlayer.hpp>

#include <iostream>

using PerfectPlayer = stochastic_nim::PerfectPlayer;
using State = PerfectPlayer::State;
using Types = PerfectPlayer::Types;

int main() {
  typename PerfectPlayer::Params params;
  PerfectPlayer player(params);
  typename PerfectPlayer::ActionResponse response = player.get_action_response(
      State(), Types::ActionMask());
  std::cout << "response: " << response.action << std::endl;

  std::cout << player.get_state_action_tensor() << std::endl;
}