#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <games/tictactoe/Constants.hpp>
#include <games/tictactoe/Game.hpp>
#include <util/BoostUtil.hpp>

#include <map>

namespace tictactoe {

class PerfectPlayer : public core::AbstractPlayer<tictactoe::Game> {
 public:
  using base_t = core::AbstractPlayer<tictactoe::Game>;

  struct Params {
    /*
     * The strength parameter controls how well the player plays. It is either 0 (random) or
     * 1 (perfect).
     */
    int strength = 1;
    bool verbose = false;
    auto make_options_description();
  };

  PerfectPlayer(const Params&);

  core::ActionResponse get_action_response(const State&, const ActionMask&) override;

 private:
  struct policy_t {
    policy_t(uint64_t u = 0);
    int select() const;
    float p[kNumCells];
  };

  using lookup_map_t = std::map<uint64_t, policy_t>;

  static lookup_map_t make_lookup_map();
  static uint64_t make_lookup(mask_t x_mask, mask_t o_mask);

  const Params params_;
  const lookup_map_t lookup_map_;
};

}  // namespace tictactoe

#include <inline/games/tictactoe/players/PerfectPlayer.inl>
