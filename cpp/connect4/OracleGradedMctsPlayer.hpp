#pragma once

/*
 * MctsPlayer that is graded by the perfect oracle.
 */

#include <common/MctsPlayer.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4PerfectPlayer.hpp>
#include <connect4/C4Tensorizor.hpp>

#include <map>
#include <mutex>

namespace c4 {

class OracleGrader {
public:
  struct mistake_tracker_t {
    float mistake_total = 0;
    float baseline_total = 0;
    int count = 0;

    void dump(const std::string descr) const;
    void update(float mistake_rate, float baseline);
  };
  using mistake_tracker_map_t = std::map<int, mistake_tracker_t>;

  OracleGrader(PerfectOracle* oracle) : oracle_(oracle) {}
  PerfectOracle* oracle() const { return oracle_; }
  void update(int score, float mistake_rate, float baseline);
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

  template<typename... BaseArgs>
  OracleGradedMctsPlayer(OracleGrader*, BaseArgs&&...);

  void start_game(common::game_id_t, const player_array_t&, common::player_index_t seat_assignment) override;
  void receive_state_change(
      common::player_index_t, const GameState&, common::action_index_t, const GameOutcome&) override;
  common::action_index_t get_action(const GameState&, const ActionMask&) override;

protected:
  void update_mistake_stats(const PerfectOracle::QueryResult& result, const GlobalPolicyProbDistr& net_policy,
                            const ActionMask& valid_actions) const;

  OracleGrader* grader_;
  PerfectOracle::MoveHistory move_history_;
};

}  // namespace c4

#include <connect4/inl/OracleGradedMctsPlayer.inl>
