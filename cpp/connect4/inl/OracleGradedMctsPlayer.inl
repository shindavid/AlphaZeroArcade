#pragma once

#include <connect4/OracleGradedMctsPlayer.hpp>

#include <iostream>

namespace c4 {

inline void OracleGrader::mistake_tracker_t::dump(const std::string descr) const {
  printf("OracleGrader %s net:%.6f mcts:%.6f baseline:%.6f\n",
         descr.c_str(), mistake_total_prior / count, mistake_total_posterior / count, baseline_total / count);
}

inline void OracleGrader::mistake_tracker_t::update(
    float mistake_rate_prior, float mistake_rate_posterior, float baseline)
{
  mistake_total_prior += mistake_rate_prior;
  mistake_total_posterior += mistake_rate_posterior;
  baseline_total += baseline;
  ++count;
}

inline void OracleGrader::update(int score, float mistake_rate_prior, float mistake_rate_posterior, float baseline) {
  std::lock_guard lock(mutex_);
  mistake_tracker_map_[score].update(mistake_rate_prior, mistake_rate_posterior, baseline);
  overall_tracker_.update(mistake_rate_prior, mistake_rate_posterior, baseline);
}

inline void OracleGrader::dump() const {
  overall_tracker_.dump("overall");
  for (const auto& [score, tracker] : mistake_tracker_map_) {
    tracker.dump(std::to_string(score));
  }
}

template<typename... BaseArgs>
OracleGradedMctsPlayer::OracleGradedMctsPlayer(OracleGrader* grader, BaseArgs&&... base_args)
    : base_t(std::forward<BaseArgs>(base_args)...), grader_(grader) {}

inline void OracleGradedMctsPlayer::start_game(
    common::game_id_t game_id, const player_array_t& players, common::player_index_t seat_assignment)
{
  move_history_.reset();
  base_t::start_game(game_id, players, seat_assignment);
}

inline void OracleGradedMctsPlayer::receive_state_change(
    common::player_index_t player, const GameState& state, common::action_index_t action, const GameOutcome& outcome)
{
  move_history_.append(action);
  base_t::receive_state_change(player, state, action, outcome);
}

inline common::action_index_t OracleGradedMctsPlayer::get_action(
    const GameState& state, const ActionMask& valid_actions)
{
  auto sim_type = this->choose_sim_type();
  const MctsResults* mcts_results = this->mcts_sim(state, sim_type);
  PerfectOracle* oracle = grader_->oracle();
  auto result = oracle->query(move_history_);
  int score = result.score;
  if (score >= 0) {  // winning or drawn position
    assert(sim_type != base_t::kRawPolicy);
    auto policy_prior = GameStateTypes::local_to_global(mcts_results->policy_prior, valid_actions);

    auto visit_counts = mcts_results->counts.cast<float>();
    auto visit_distr = visit_counts / visit_counts.sum();
    update_mistake_stats(result, policy_prior, visit_distr, valid_actions);
  }

  return this->get_action_helper(sim_type, mcts_results, valid_actions);
}

inline void OracleGradedMctsPlayer::update_mistake_stats(
    const PerfectOracle::QueryResult& result, const GlobalPolicyProbDistr& net_policy,
    const GlobalPolicyProbDistr& posterior_policy, const ActionMask& valid_actions) const
{
  int score = result.score;
  auto correct_moves = result.good_moves;

  float mistake_prior_num = 0;
  float mistake_prior_den = 0;

  float mistake_posterior_num = 0;
  float mistake_posterior_den = 0;

  int baseline_num = 0;
  int baseline_den = 0;

  for (int i = 0; i < kNumColumns; ++i) {
    if (!valid_actions[i]) continue;
    float prior = net_policy(i);
    float posterior = posterior_policy(i);

    mistake_prior_den += prior;
    mistake_posterior_den += posterior;
    baseline_den++;
    if (!correct_moves[i]) {
      mistake_prior_num += prior;
      mistake_posterior_num += posterior;
      baseline_num++;
    }
  }

  float mistake_prior_rate = mistake_prior_den ? mistake_prior_num / mistake_prior_den : 0;
  float mistake_mcts_rate = mistake_posterior_den ? mistake_posterior_num / mistake_posterior_den : 0;
  float baseline = baseline_num * 1.0 / baseline_den;

  grader_->update(score, mistake_prior_rate, mistake_mcts_rate, baseline);
}

}  // namespace c4
