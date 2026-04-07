#include "generic_players/alpha0/Player.hpp"

#include "core/Constants.hpp"
#include "search/VerboseManager.hpp"

#include <unistd.h>

namespace generic::alpha0 {

template <search::concepts::SearchSpec SearchSpec>
auto Player<SearchSpec>::Params::make_options_description() {
  namespace po = boost::program_options;

  auto desc = BaseParams::make_options_description();

  return desc
    .template add_option<"lcb-z-score">(
      po::value<float>(&this->LCB_z_score)->default_value(this->LCB_z_score),
      "z-score for LCB. If zero, disable LCB")
    .template add_option<"verbose-num-rows-to-display", 'r'>(
      po::value<int>(&this->verbose_num_rows_to_display)
        ->default_value(this->verbose_num_rows_to_display),
      "MCTS player number of rows to display in verbose mode");
}

template <search::concepts::SearchSpec SearchSpec>
void Player<SearchSpec>::receive_state_change(const StateChangeUpdate& update) {
  Base::receive_state_change(update);

  if (this->get_my_seat() == update.seat() && this->verbose()) {
    auto it = update.state_it();

    if (VerboseManager::get_instance()->auto_terminal_printing_enabled()) {
      Game::IO::print_state(std::cout, it->state, update.move(), &this->get_player_names());
    }

    if (it->aux) {
      AuxData* aux_data = reinterpret_cast<AuxData*>(it->aux);
      VerboseManager::get_instance()->set(aux_data->verbose_data);
    }
  }
}

template <search::concepts::SearchSpec SearchSpec>
typename Player<SearchSpec>::ActionResponse Player<SearchSpec>::get_action_response_helper(
  const SearchResults* mcts_results, const ActionRequest& request) {
  PolicyTensor modified_policy = get_action_policy(mcts_results, request.valid_moves);
  ActionResponse action_response(
    PolicyEncoding::to_move(request.state, eigen_util::sample(modified_policy)));

  if (this->verbose() || this->is_facing_backtracking_opponent()) {
    if (this->is_facing_backtracking_opponent() || this->aux_data_ptrs_.empty()) {
      this->aux_data_ptrs_.push_back(new AuxData(action_response));
    }
    AuxData* aux_data = this->aux_data_ptrs_.back();
    if (this->verbose()) {
      aux_data->verbose_data = std::make_shared<VerboseData>(
        modified_policy, *mcts_results, params_extra_.verbose_num_rows_to_display);
      VerboseManager::get_instance()->set(aux_data->verbose_data);
    }
    action_response.set_aux(aux_data);
  }
  return action_response;
}

template <search::concepts::SearchSpec SearchSpec>
typename Player<SearchSpec>::PolicyTensor Player<SearchSpec>::get_action_policy(
  const SearchResults* mcts_results, const MoveSet& valid_moves) const {
  PolicyTensor policy;
  const auto& frame = mcts_results->frame;
  const auto& counts = mcts_results->counts;
  if (this->search_mode_ == core::kRawPolicy) {
    this->raw_init(mcts_results, valid_moves, policy);
  } else if (this->search_params_[this->search_mode_].tree_size_limit <= 1) {
    policy = mcts_results->P;
  } else {
    policy = counts;
  }

  if (this->search_mode_ != core::kRawPolicy &&
      this->search_params_[this->search_mode_].tree_size_limit > 1) {
    policy = mcts_results->action_symmetry_table.collapse(frame, policy);
    this->apply_temperature(policy);
    policy = mcts_results->action_symmetry_table.symmetrize(frame, policy);
    if (params_extra_.LCB_z_score) {
      apply_LCB(mcts_results, valid_moves, policy);
    }
  }

  this->normalize(frame, valid_moves, policy);
  return policy;
}

template <search::concepts::SearchSpec SearchSpec>
void Player<SearchSpec>::apply_LCB(const SearchResults* mcts_results, const MoveSet& valid_moves,
                                   PolicyTensor& policy) const {
  const auto& table = mcts_results->action_symmetry_table;
  const auto& counts = mcts_results->counts;
  const auto& frame = mcts_results->frame;

  PolicyTensor Q_sum = table.collapse(frame, mcts_results->AQs) * policy;
  PolicyTensor Q_sq_sum = table.collapse(frame, mcts_results->AQs_sq) * policy;

  Q_sum = table.symmetrize(frame, Q_sum);
  Q_sq_sum = table.symmetrize(frame, Q_sq_sum);

  PolicyTensor Q = Q_sum / counts;
  PolicyTensor Q_sq = Q_sq_sum / counts;
  PolicyTensor Q_sigma_sq = (Q_sq - Q * Q) / counts;
  Q_sigma_sq = eigen_util::cwiseMax(Q_sigma_sq, 0);  // clip negative values to 0
  PolicyTensor Q_sigma = Q_sigma_sq.sqrt();

  PolicyTensor LCB = Q - params_extra_.LCB_z_score * Q_sigma;

  float policy_max = -1;
  float min_LCB = 0;
  bool min_LCB_set = false;

  // Let S be the set of indices at which policy is maximal. The below loop sets min_LCB to
  // min_{i in S} {LCB(i)}
  for (Move move : valid_moves) {
    auto index = PolicyEncoding::to_index(frame, move);
    float p = policy.coeff(index);
    if (p <= 0) continue;

    if (p > policy_max) {
      policy_max = p;
      min_LCB = LCB.coeff(index);
      min_LCB_set = true;
    } else if (p == policy_max) {
      min_LCB = std::min(min_LCB, LCB.coeff(index));
      min_LCB_set = true;
    }
  }

  if (min_LCB_set) {
    PolicyTensor UCB = Q + params_extra_.LCB_z_score * Q_sigma;

    // zero out policy wherever UCB < min_LCB
    auto mask = (UCB >= min_LCB).template cast<float>();
    PolicyTensor policy_masked = policy * mask;

    if (search::kEnableSearchDebug) {
      int visited_actions = 0;
      for (Move move : valid_moves) {
        auto index = PolicyEncoding::to_index(frame, move);
        if (counts.coeff(index)) visited_actions++;
      }

      LocalPolicyArray counts_arr(visited_actions);
      LocalPolicyArray policy_arr(visited_actions);
      LocalPolicyArray Q_arr(visited_actions);
      LocalPolicyArray Q_sigma_arr(visited_actions);
      LocalPolicyArray LCB_arr(visited_actions);
      LocalPolicyArray UCB_arr(visited_actions);
      LocalPolicyArray policy_masked_arr(visited_actions);

      int r = 0;
      for (Move move : valid_moves) {
        auto index = PolicyEncoding::to_index(frame, move);
        if (counts.coeff(index) == 0) continue;

        counts_arr(r) = counts.coeff(index);
        policy_arr(r) = policy.coeff(index);
        Q_arr(r) = Q.coeff(index);
        Q_sigma_arr(r) = Q_sigma.coeff(index);
        LCB_arr(r) = LCB.coeff(index);
        UCB_arr(r) = UCB.coeff(index);
        policy_masked_arr(r) = policy_masked.coeff(index);

        r++;
      }

      ActionPrinter printer(valid_moves);
      LocalPolicyArray actions_arr = printer.flat_array();

      policy_arr /= policy_arr.sum();
      policy_masked_arr /= policy_masked_arr.sum();

      static std::vector<std::string> columns = {"action",  "N",   "P",   "Q",
                                                 "Q_sigma", "LCB", "UCB", "P*"};
      auto data = eigen_util::sort_rows(
        eigen_util::concatenate_columns(actions_arr, counts_arr, policy_arr, Q_arr, Q_sigma_arr,
                                        LCB_arr, UCB_arr, policy_masked_arr));

      eigen_util::PrintArrayFormatMap fmt_map;
      printer.update_format_map(fmt_map);

      std::cout << std::endl << "Applying LCB:" << std::endl;
      eigen_util::print_array(std::cout, data, columns, &fmt_map);
    }

    policy = policy_masked;
  }
}

}  // namespace generic::alpha0
