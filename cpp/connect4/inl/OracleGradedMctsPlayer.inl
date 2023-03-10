#pragma once

#include <connect4/OracleGradedMctsPlayer.hpp>

#include <iostream>

namespace c4 {

inline void OracleGrader::mistake_tracker_t::dump(const std::string descr) const {
  printf("OracleGrader %s count:%d net-t0:%.6f net-t1:%.6f mcts-t0:%.6f mcts-t1:%.6f baseline:%.6f\n",
         descr.c_str(), count, mistake_total_prior_t0 / count, mistake_total_prior_t1 / count,
         mistake_total_posterior_t0 / count, mistake_total_posterior_t1 / count, baseline_total / count);
}

inline void OracleGrader::mistake_tracker_t::update(
    float mistake_rate_prior_t0, float mistake_rate_prior_t1,
    float mistake_rate_posterior_t0, float mistake_rate_posterior_t1, float baseline)
{
  mistake_total_prior_t0 += mistake_rate_prior_t0;
  mistake_total_prior_t1 += mistake_rate_prior_t1;
  mistake_total_posterior_t0 += mistake_rate_posterior_t0;
  mistake_total_posterior_t1 += mistake_rate_posterior_t1;
  baseline_total += baseline;
  ++count;
}

inline void OracleGrader::update(int score, int move_number, float mistake_rate_prior_t0, float mistake_rate_prior_t1,
                                 float mistake_rate_posterior_t0, float mistake_rate_posterior_t1, float baseline)
{
  std::lock_guard lock(mutex_);
  auto key = std::make_pair(move_number, score);
  mistake_tracker_map_[key].update(mistake_rate_prior_t0, mistake_rate_prior_t1,
                                   mistake_rate_posterior_t0, mistake_rate_posterior_t1, baseline);
  overall_tracker_.update(mistake_rate_prior_t0, mistake_rate_prior_t1,
                          mistake_rate_posterior_t0, mistake_rate_posterior_t1, baseline);
}

inline void OracleGrader::dump() const {
  overall_tracker_.dump("overall");
  for (const auto& [key, tracker] : mistake_tracker_map_) {
    int move_number = key.first;
    int score = key.second;
    std::string descr = util::create_string("%d-%d", move_number, score);
    tracker.dump(descr);
  }
  printf("OracleGrader done\n");
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
  if (sim_type != kRawPolicy) {
    PerfectOracle *oracle = grader_->oracle();
    auto result = oracle->query(move_history_);
    if (result.score >= 0) {  // winning or drawn position
      assert(sim_type != base_t::kRawPolicy);
      auto policy_prior = GameStateTypes::local_to_global(mcts_results->policy_prior, valid_actions);

      auto visit_counts = mcts_results->counts.cast<float>();
      auto visit_distr = visit_counts / visit_counts.sum();
      update_mistake_stats(result, policy_prior, visit_distr, valid_actions, state.get_move_number());
    }
  }
  return this->get_action_helper(sim_type, mcts_results, valid_actions);
}

inline void OracleGradedMctsPlayer::update_mistake_stats(
    const PerfectOracle::QueryResult& result, const GlobalPolicyProbDistr& net_policy_t1,
    const GlobalPolicyProbDistr& posterior_policy_t1, const ActionMask& valid_actions, int move_number) const
{
  GlobalPolicyProbDistr net_policy_t0 = (net_policy_t1 == net_policy_t1.maxCoeff()).template cast<float>();
  GlobalPolicyProbDistr posterior_policy_t0 = (posterior_policy_t1 == posterior_policy_t1.maxCoeff()).template cast<float>();

  net_policy_t0 /= net_policy_t0.sum();
  posterior_policy_t0 /= posterior_policy_t0.sum();

  int score = result.score;
  auto correct_moves = result.good_moves;

  float mistake_prior_t0_num = 0;
  float mistake_prior_t0_den = 0;

  float mistake_prior_t1_num = 0;
  float mistake_prior_t1_den = 0;

  float mistake_posterior_t0_num = 0;
  float mistake_posterior_t0_den = 0;

  float mistake_posterior_t1_num = 0;
  float mistake_posterior_t1_den = 0;

  int baseline_num = 0;
  int baseline_den = 0;

  for (int i = 0; i < kNumColumns; ++i) {
    if (!valid_actions[i]) continue;

    float prior_t0 = net_policy_t0(i);
    float posterior_t0 = posterior_policy_t0(i);

    float prior_t1 = net_policy_t1(i);
    float posterior_t1 = posterior_policy_t1(i);

    mistake_prior_t0_den += prior_t0;
    mistake_posterior_t0_den += posterior_t0;
    mistake_prior_t1_den += prior_t1;
    mistake_posterior_t1_den += posterior_t1;
    baseline_den++;
    if (!correct_moves[i]) {
      mistake_prior_t0_num += prior_t0;
      mistake_posterior_t0_num += posterior_t0;
      mistake_prior_t1_num += prior_t1;
      mistake_posterior_t1_num += posterior_t1;
      baseline_num++;
    }
  }

  float mistake_prior_t0_rate = mistake_prior_t0_den ? mistake_prior_t0_num / mistake_prior_t0_den : 0;
  float mistake_mcts_t0_rate = mistake_posterior_t0_den ? mistake_posterior_t0_num / mistake_posterior_t0_den : 0;
  float mistake_prior_t1_rate = mistake_prior_t1_den ? mistake_prior_t1_num / mistake_prior_t1_den : 0;
  float mistake_mcts_t1_rate = mistake_posterior_t1_den ? mistake_posterior_t1_num / mistake_posterior_t1_den : 0;
  float baseline = baseline_num * 1.0 / baseline_den;

  grader_->update(score, move_number, mistake_prior_t0_rate, mistake_prior_t1_rate,
                  mistake_mcts_t0_rate, mistake_mcts_t1_rate, baseline);
}

}  // namespace c4
