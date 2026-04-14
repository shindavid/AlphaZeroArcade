#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/OraclePool.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/StockfishProcess.hpp"

namespace a0achess {

class StockfishPlayer : public core::AbstractPlayer<Game> {
 public:
  using StockfishPool = core::OraclePool<StockfishProcess>;
  using ActionResponse = core::ActionResponse<Game>;

  struct Params {
    int depth = 10;
    int num_stockfish_procs = 8;
    auto make_options_description();
  };

  StockfishPlayer(StockfishPool* stockfish_pool, const Params& params)
      : stockfish_pool_(stockfish_pool), params_(params) {}

  ActionResponse get_action_response(const ActionRequest& request) override;

  bool start_game() override {
    move_value_history_.clear();
    return true;
  }

  void receive_state_change(const StateChangeUpdate& update) override {
    move_value_history_.push_back(update.move()->move());
  }

  State compute_state() const {
    State state;
    Game::Rules::init_state(state);
    for (auto v : move_value_history_) {
      Move m = Move(v);
      Game::Rules::apply(state, m);
    }
    return state;
  }

  std::string get_fen_move() const {
    std::string move_strs;
    for (auto v : move_value_history_) {
      Move m = Move(v);
      move_strs += " " + m.to_str();
    }
    return move_strs;
  }

 private:
  StockfishPool* const stockfish_pool_;
  Params params_;

  std::vector<int16_t> move_value_history_;
};

}  // namespace a0achess

#include "inline/games/chess/players/StockfishPlayer.inl"
