#include "generic_players/alpha0/VerboseData.hpp"
#include "util/EigenUtil.hpp"

namespace generic::alpha0 {

template <search::concepts::Traits Traits>
auto VerboseData<Traits>::build_action_data() const {
  const auto& valid_actions = mcts_results.valid_actions;
  const auto& mcts_counts = mcts_results.counts;
  const auto& net_policy = mcts_results.policy_prior;

  int num_valid = valid_actions.count();

  // Zero() calls: not necessary, but silences gcc warning, and is cheap enough
  LocalPolicyArray actions_arr = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray net_policy_arr = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray action_policy_arr = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray mcts_counts_arr = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray posterior_arr = LocalPolicyArray::Zero(num_valid);

  float total_count = 0;
  for (int a : valid_actions.on_indices()) {
    total_count += mcts_counts(a);
  }

  int r = 0;
  for (int a : valid_actions.on_indices()) {
    actions_arr(r) = a;
    net_policy_arr(r) = net_policy(a);
    action_policy_arr(r) = action_policy(a);
    mcts_counts_arr(r) = mcts_counts(a);
    r++;
  }

  posterior_arr = mcts_counts_arr / total_count;

  auto data = eigen_util::sort_rows(
    eigen_util::concatenate_columns(actions_arr, net_policy_arr, posterior_arr, mcts_counts_arr,
                                    action_policy_arr),
    3, false);

  return data;
}

static std::vector<std::string>& get_column_names() {
  static std::vector<std::string> columns = {"action", "Prior", "Posterior", "Counts", "Modified"};
  return columns;
}

template <search::concepts::Traits Traits>
boost::json::object VerboseData<Traits>::to_json() const {
  const auto& win_rates = mcts_results.win_rates;
  const auto& net_value = mcts_results.value_prior;
  core::action_mode_t action_mode = mcts_results.action_mode;

  eigen_util::PrintArrayFormatMap fmt_map{
    {"Player", [&](core::seat_index_t x) { return IO::player_to_str(x); }},
    {"action", [&](float x) { return IO::action_to_str(x, action_mode); }},
  };

  boost::json::object obj;
  boost::json::object cpu_eval = Game::GameResults::to_json(net_value, win_rates, &fmt_map);
  obj["cpu_pos_eval"] = std::move(cpu_eval);

  if (Game::Rules::is_chance_mode(action_mode)) return obj;

  auto data = build_action_data();
  auto columns = get_column_names();
  obj["actions"] = eigen_util::output_to_json(data, columns, &fmt_map);
  return obj;
}

template <search::concepts::Traits Traits>
void VerboseData<Traits>::to_terminal_text() const {
  std::cout << std::endl << "CPU pos eval:" << std::endl;
  const auto& valid_actions = mcts_results.valid_actions;
  const auto& win_rates = mcts_results.win_rates;
  const auto& net_value = mcts_results.value_prior;
  core::action_mode_t action_mode = mcts_results.action_mode;

  eigen_util::PrintArrayFormatMap fmt_map{
    {"Player", [&](core::seat_index_t x) { return IO::player_to_str(x); }},
    {"action", [&](float x) { return IO::action_to_str(x, action_mode); }},
  };

  Game::GameResults::print_array(net_value, win_rates, &fmt_map);

  if (Game::Rules::is_chance_mode(mcts_results.action_mode)) return;

  int num_valid = valid_actions.count();
  int num_rows = std::min(num_valid, n_rows_to_display_);
  auto data = build_action_data();
  auto columns = get_column_names();

  eigen_util::print_array(std::cout, data.topRows(num_rows), columns, &fmt_map);

  if (num_valid > num_rows) {
    int x = num_valid - num_rows;
    if (x == 1) {
      std::cout << "... 1 row not displayed" << std::endl;
    } else {
      std::cout << "... " << x << " rows not displayed" << std::endl;
    }
  } else {
    for (int i = 0; i < n_rows_to_display_ - num_rows + 1; i++) {
      std::cout << std::endl;
    }
  }
  std::cout << "******************************" << std::endl;
}

template <search::concepts::Traits Traits>
void VerboseData<Traits>::set(const PolicyTensor& policy, const SearchResults& results) {
  action_policy = policy;
  mcts_results = results;
}

}  // namespace generic::alpha0
