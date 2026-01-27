#include "generic_players/beta0/Player.hpp"

#include "search/VerboseManager.hpp"
#include "util/EigenUtil.hpp"

#include <string>
#include <vector>

namespace generic::beta0 {

template <search::concepts::Traits Traits>
typename Player<Traits>::PolicyTensor Player<Traits>::get_action_policy(
  const SearchResults* mcts_results, const ActionMask& valid_actions) const {
  PolicyTensor policy;
  if (this->search_mode_ == core::kRawPolicy) {
    this->raw_init(mcts_results, valid_actions, policy);
    this->normalize(valid_actions, policy);
  } else {
    PolicyTensor collapsed_policy = mcts_results->action_symmetry_table.collapse(mcts_results->pi);
    PolicyTensor temped_policy = collapsed_policy;
    this->apply_temperature(temped_policy);
    PolicyTensor symmed_policy = mcts_results->action_symmetry_table.symmetrize(temped_policy);
    policy = symmed_policy;
    this->normalize(valid_actions, policy);

    // TODO: LCB method
    if (search::kEnableSearchDebug) {
      int n = valid_actions.count();

      LocalPolicyArray actions_arr(n);
      LocalPolicyArray P(n);
      LocalPolicyArray pi(n);
      LocalPolicyArray collapsed_pi(n);
      LocalPolicyArray temped_pi(n);
      LocalPolicyArray symmed_pi(n);
      LocalPolicyArray final_pi(n);

      int r = 0;
      for (int a : valid_actions.on_indices()) {
        actions_arr(r) = a;
        P(r) = mcts_results->P(a);
        pi(r) = mcts_results->pi(a);
        collapsed_pi(r) = collapsed_policy(a);
        temped_pi(r) = temped_policy(a);
        symmed_pi(r) = symmed_policy(a);
        final_pi(r) = policy(a);

        r++;
      }

      static std::vector<std::string> columns = {"action", "P",    "pi", "c_pi",
                                                 "t_pi",   "s_pi", "pi*"};
      auto data = eigen_util::sort_rows(eigen_util::concatenate_columns(
        actions_arr, P, pi, collapsed_pi, temped_pi, symmed_pi, final_pi));

      core::action_mode_t mode = mcts_results->action_mode;
      eigen_util::PrintArrayFormatMap fmt_map{
        {"action", [&](float x) { return Game::IO::action_to_str(x, mode); }},
      };

      std::cout << "Action selection:" << std::endl;
      eigen_util::print_array(std::cout, data, columns, &fmt_map);
      std::cout << std::endl;
    }

    policy = symmed_policy;
  }

  return policy;
}

template <search::concepts::Traits Traits>
auto Player<Traits>::Params::make_options_description() {
  namespace po = boost::program_options;

  auto desc = BaseParams::make_options_description();

  return desc.template add_option<"verbose", 'v'>(
    po::bool_switch(&this->verbose)->default_value(this->verbose), "MCTS player verbose mode");
}

template <search::concepts::Traits Traits>
core::ActionResponse Player<Traits>::get_action_response_helper(const SearchResults* mcts_results,
                                                                const ActionRequest& request) {
  PolicyTensor modified_policy = get_action_policy(mcts_results, request.valid_actions);
  core::ActionResponse action_response = eigen_util::sample(modified_policy);

  if (params_extra_.verbose || this->is_facing_backtracking_opponent()) {
    if (this->is_facing_backtracking_opponent() || this->aux_data_ptrs_.empty()) {
      this->aux_data_ptrs_.push_back(new AuxData(action_response));
    }
    AuxData* aux_data = this->aux_data_ptrs_.back();
    if (params_extra_.verbose) {
      aux_data->verbose_data = std::make_shared<VerboseData>(
        modified_policy, *mcts_results);
      VerboseManager::get_instance()->set(aux_data->verbose_data);
    }
    action_response.set_aux(aux_data);
  }
  return action_response;
}


}  // namespace generic::beta0
