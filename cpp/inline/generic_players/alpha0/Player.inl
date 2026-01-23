#include "generic_players/alpha0/Player.hpp"

#include "core/Constants.hpp"
#include "search/VerboseManager.hpp"

#include <unistd.h>

namespace generic::alpha0 {

template <search::concepts::Traits Traits>
auto Player<Traits>::Params::make_options_description() {
  namespace po = boost::program_options;

  auto desc = BaseParams::make_options_description();

  return desc
    .template add_option<"lcb-z-score">(
      po::value<float>(&this->LCB_z_score)->default_value(this->LCB_z_score),
      "z-score for LCB. If zero, disable LCB")
    .template add_option<"verbose", 'v'>(
      po::bool_switch(&this->verbose)->default_value(this->verbose), "MCTS player verbose mode")
    .template add_option<"verbose-num-rows-to-display", 'r'>(
      po::value<int>(&this->verbose_num_rows_to_display)
        ->default_value(this->verbose_num_rows_to_display),
      "MCTS player number of rows to display in verbose mode");
}

template <search::concepts::Traits Traits>
Player<Traits>::Player(const Params& params, SharedData_sptr shared_data, bool owns_shared_data)
    : Base(params, shared_data, owns_shared_data), params_extra_(params) {
  if (params.verbose) {
    verbose_info_ = new VerboseData<Traits>(params.verbose_num_rows_to_display);
  }
}

template <search::concepts::Traits Traits>
Player<Traits>::~Player() {
  if (verbose_info_) {
    delete verbose_info_;
  }
}

template <search::concepts::Traits Traits>
void Player<Traits>::receive_state_change(const StateChangeUpdate& update) {
  Base::receive_state_change(update);

  if (this->get_my_seat() == update.seat() && params_extra_.verbose) {
    const State& state = (*update.state_it()).state;
    if (VerboseManager::get_instance()->auto_terminal_printing_enabled()) {
      Game::IO::print_state(std::cout, state, update.action(),
                            &this->get_player_names());
    }
    auto aux = (*update.state_it()).aux;
    if (aux) {
      SearchResults* mcts_results = reinterpret_cast<SearchResults*>(aux);
      ActionMask valid_actions = Game::Rules::get_legal_moves(state);
      PolicyTensor modified_policy = get_action_policy(mcts_results, valid_actions);
      verbose_info_->set(modified_policy, *mcts_results);
      VerboseManager::get_instance()->set(verbose_info_);
    }
  }
}

template <search::concepts::Traits Traits>
core::ActionResponse Player<Traits>::get_action_response_helper(const SearchResults* mcts_results,
                                                                const ActionRequest& request) {
  PolicyTensor modified_policy = get_action_policy(mcts_results, request.valid_actions);

  if (verbose_info_) {
    verbose_info_->set(modified_policy, *mcts_results);
    VerboseManager::get_instance()->set(verbose_info_);
  }

  return eigen_util::sample(modified_policy);
}

template <search::concepts::Traits Traits>
typename Player<Traits>::PolicyTensor Player<Traits>::get_action_policy(
  const SearchResults* mcts_results, const ActionMask& valid_actions) const {
  PolicyTensor policy;
  const auto& counts = mcts_results->counts;
  if (this->search_mode_ == core::kRawPolicy) {
    this->raw_init(mcts_results, valid_actions, policy);
  } else if (this->search_params_[this->search_mode_].tree_size_limit <= 1) {
    policy = mcts_results->P;
  } else {
    policy = counts;
  }

  if (this->search_mode_ != core::kRawPolicy &&
      this->search_params_[this->search_mode_].tree_size_limit > 1) {
    policy = mcts_results->action_symmetry_table.collapse(policy);
    this->apply_temperature(policy);
    policy = mcts_results->action_symmetry_table.symmetrize(policy);
    if (params_extra_.LCB_z_score) {
      apply_LCB(mcts_results, valid_actions, policy);
    }
  }

  this->normalize(valid_actions, policy);
  return policy;
}

template <search::concepts::Traits Traits>
void Player<Traits>::apply_LCB(const SearchResults* mcts_results, const ActionMask& valid_actions,
                               PolicyTensor& policy) const {
  const auto& counts = mcts_results->counts;

  PolicyTensor Q_sum = mcts_results->action_symmetry_table.collapse(mcts_results->AQs) * policy;
  PolicyTensor Q_sq_sum =
    mcts_results->action_symmetry_table.collapse(mcts_results->AQs_sq) * policy;

  Q_sum = mcts_results->action_symmetry_table.symmetrize(Q_sum);
  Q_sq_sum = mcts_results->action_symmetry_table.symmetrize(Q_sq_sum);

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
  for (int a : valid_actions.on_indices()) {
    float p = policy(a);
    if (p <= 0) continue;

    if (p > policy_max) {
      policy_max = p;
      min_LCB = LCB(a);
      min_LCB_set = true;
    } else if (p == policy_max) {
      min_LCB = std::min(min_LCB, LCB(a));
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
      for (int a : valid_actions.on_indices()) {
        if (counts(a)) visited_actions++;
      }

      LocalPolicyArray actions_arr(visited_actions);
      LocalPolicyArray counts_arr(visited_actions);
      LocalPolicyArray policy_arr(visited_actions);
      LocalPolicyArray Q_arr(visited_actions);
      LocalPolicyArray Q_sigma_arr(visited_actions);
      LocalPolicyArray LCB_arr(visited_actions);
      LocalPolicyArray UCB_arr(visited_actions);
      LocalPolicyArray policy_masked_arr(visited_actions);

      int r = 0;
      for (int a : valid_actions.on_indices()) {
        if (counts(a) == 0) continue;

        actions_arr(r) = a;
        counts_arr(r) = counts(a);
        policy_arr(r) = policy(a);
        Q_arr(r) = Q(a);
        Q_sigma_arr(r) = Q_sigma(a);
        LCB_arr(r) = LCB(a);
        UCB_arr(r) = UCB(a);
        policy_masked_arr(r) = policy_masked(a);

        r++;
      }

      policy_arr /= policy_arr.sum();
      policy_masked_arr /= policy_masked_arr.sum();

      static std::vector<std::string> columns = {"action",  "N",   "P",   "Q",
                                                 "Q_sigma", "LCB", "UCB", "P*"};
      auto data = eigen_util::sort_rows(
        eigen_util::concatenate_columns(actions_arr, counts_arr, policy_arr, Q_arr, Q_sigma_arr,
                                        LCB_arr, UCB_arr, policy_masked_arr));

      core::action_mode_t mode = mcts_results->action_mode;
      eigen_util::PrintArrayFormatMap fmt_map{
        {"action", [&](float x) { return Game::IO::action_to_str(x, mode); }},
      };

      std::cout << std::endl << "Applying LCB:" << std::endl;
      eigen_util::print_array(std::cout, data, columns, &fmt_map);
    }

    policy = policy_masked;
  }
}

}  // namespace generic::alpha0
