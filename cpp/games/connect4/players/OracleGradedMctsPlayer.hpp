#pragma once

/*
 * MctsPlayer that is graded by the perfect oracle.
 */

#include <common/players/MctsPlayer.hpp>
#include <games/connect4/GameState.hpp>
#include <games/connect4/players/PerfectPlayer.hpp>
#include <games/connect4/Tensorizor.hpp>

#include <map>
#include <mutex>

namespace c4 {

class OracleGrader {
public:
  struct mistake_tracker_t {
    float mistake_total_prior_t0 = 0;
    float mistake_total_prior_t1 = 0;
    float mistake_total_posterior_t0 = 0;
    float mistake_total_posterior_t1 = 0;
    float baseline_total = 0;
    int count = 0;

    void dump(const std::string descr) const;
    void update(float mistake_rate_prior_t0, float mistake_rate_prior_t1,
                float mistake_rate_posterior_t0, float mistake_rate_posterior_t1, float baseline);
  };
  using key_t = std::pair<int, int>;  // (move-number, score)
  using mistake_tracker_map_t = std::map<key_t, mistake_tracker_t>;

  OracleGrader(PerfectOracle* oracle) : oracle_(oracle) {}
  PerfectOracle* oracle() const { return oracle_; }
  void update(int score, int move_number, float mistake_rate_prior_t0, float mistake_rate_prior_t1,
              float mistake_rate_posterior_t0, float mistake_rate_posterior_t1, float baseline);
  void dump() const;

protected:
  std::mutex mutex_;
  PerfectOracle* oracle_;
  mistake_tracker_map_t mistake_tracker_map_;
  mistake_tracker_t overall_tracker_;
};

class OracleGradedMctsPlayer : public common::MctsPlayer<GameState, Tensorizor> {
public:
  using base_t = common::MctsPlayer<GameState, Tensorizor>;
  using GameStateTypes = core::GameStateTypes<GameState>;
  using PolicyArray = typename GameStateTypes::PolicyArray;

  template<typename... BaseArgs>
  OracleGradedMctsPlayer(OracleGrader*, BaseArgs&&...);

  void start_game() override;
  void receive_state_change(
      core::seat_index_t, const GameState&, core::action_index_t) override;
  core::action_index_t get_action(const GameState&, const ActionMask&) override;

protected:
  void update_mistake_stats(const PerfectOracle::QueryResult& result, const PolicyArray& net_policy,
                            const PolicyArray& posterior_policy, const ActionMask& valid_actions,
                            int move_number) const;

  OracleGrader* grader_;
  PerfectOracle::MoveHistory move_history_;
};

}  // namespace c4

#include <games/connect4/players/inl/OracleGradedMctsPlayer.inl>
