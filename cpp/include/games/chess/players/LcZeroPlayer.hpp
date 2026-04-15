#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/OraclePool.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/LcZeroProcess.hpp"

namespace a0achess {

class LcZeroPlayer : public core::AbstractPlayer<Game> {
 public:
  using LcZeroPool = core::OraclePool<LcZeroProcess>;
  using ActionResponse = core::ActionResponse<Game>;

  struct Params {
    using pair_t = std::pair<std::string_view, int Params::*>;

    int num_procs = 5;
    int movetime = -1;
    int nodes = 1200;

    static constexpr std::array<pair_t, 2> go_options = {
      {{"movetime", &Params::movetime}, {"nodes", &Params::nodes}}};

    auto make_options_description();
    std::string build_go_command() const;
  };

  LcZeroPlayer(LcZeroPool* lc0_pool, const Params& params)
      : lc0_pool_(lc0_pool), params_(params) {}

  ActionResponse get_action_response(const ActionRequest& request) override;

  bool start_game() override {
    move_value_history_.clear();
    return true;
  }

  void receive_state_change(const StateChangeUpdate& update) override {
    move_value_history_.push_back(update.move()->move());
  }

  std::string get_fen_move() const;

 private:
  LcZeroPool* const lc0_pool_;
  Params params_;

  std::vector<int16_t> move_value_history_;
};

}  // namespace a0achess

#include "inline/games/chess/players/LcZeroPlayer.inl"
