#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/OraclePool.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/UciProcess.hpp"

namespace a0achess {

class UciPlayer : public core::AbstractPlayer<Game> {
 public:
  using UciPool = core::OraclePool<UciProcess>;
  using ActionResponse = core::ActionResponse<Game>;
  using ProcParams = UciProcess::Params;

  struct Params {
    using pair_t = std::pair<std::string_view, int Params::*>;

    int num_procs = -1;
    int movetime = -1;
    int depth = -1;
    int nodes = -1;

    static constexpr std::array<pair_t, 3> go_options = {
      {{"movetime", &Params::movetime}, {"nodes", &Params::nodes}, {"depth", &Params::depth}}};

    auto make_options_description();
    std::string build_go_command() const;
  };

  UciPlayer(UciPool* pool, const Params& params, const ProcParams& proc_params)
      : pool_(pool), params_(params), proc_params_(proc_params) {}

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
  UciPool* const pool_;
  Params params_;
  ProcParams proc_params_;

  std::vector<int16_t> move_value_history_;
};

}  // namespace a0achess

#include "inline/games/chess/players/UciPlayer.inl"
