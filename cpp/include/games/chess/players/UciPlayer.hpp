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
      : pool_(pool), proc_params_(proc_params), go_cmd_(params.build_go_command()) {}

  ActionResponse get_action_response(const ActionRequest& request) override;

  bool start_game() override {
    move_str_.clear();
    return true;
  }

  void receive_state_change(const StateChangeUpdate& update) override {
    if (!update.is_jump()) {
      move_str_ += " " + update.move()->to_str();
    } else {
      move_str_.clear();
      auto state_it = update.state_it();

      std::vector<const Move*> moves;
      moves.reserve(state_it->step);

      while (!state_it.end()) {
        moves.push_back(&state_it->move_from_parent);
        ++state_it;
      }
      for (auto it = moves.rbegin(); it != moves.rend(); ++it) {
        move_str_ += ' ';
        move_str_ += (*it)->to_str();
      }
    }
  }

 private:
  UciPool* const pool_;
  ProcParams proc_params_;
  std::string go_cmd_;
  std::string move_str_;
};

}  // namespace a0achess

#include "inline/games/chess/players/UciPlayer.inl"
