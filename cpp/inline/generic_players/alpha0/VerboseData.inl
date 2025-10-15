#include "generic_players/alpha0/VerboseData.hpp"

namespace generic::alpha0 {

template <search::concepts::Traits Traits>
typename VerboseData<Traits>::Table VerboseData<Traits>::build_table_(int n_rows_to_display) const {
  const auto& valid_actions = mcts_results.valid_actions;
  const auto& mcts_counts = mcts_results.counts;
  const auto& net_policy = mcts_results.policy_prior;
  const auto& win_rates = mcts_results.win_rates;
  const auto& net_value = mcts_results.value_prior;
  const auto action_mode = mcts_results.action_mode;

  Table table;
  table.action_mode = action_mode;

  // flatten net_value and win_rates (assumed Eigen arrays) to std::vector<float>
  table.net_value_v.assign(net_value.data(), net_value.data() + net_value.size());
  table.win_rates_v.assign(win_rates.data(), win_rates.data() + win_rates.size());

  if (Game::Rules::is_chance_mode(action_mode)) {
    table.rows_sorted.clear();
    table.omitted_rows = 0;
    return table;
  }

  int num_valid = valid_actions.count();
  table.rows_sorted.reserve(num_valid);

  float total_count = 0.0f;
  for (int a : valid_actions.on_indices()) total_count += mcts_counts(a);
  const float denom = total_count > 0.0f ? total_count : 1.0f;

  for (int a : valid_actions.on_indices()) {
    VerboseRow r;
    r.action = a;
    r.prior = net_policy(a);
    r.posterior = mcts_counts(a) / denom;
    r.counts = mcts_counts(a);
    r.modified = action_policy(a);
    table.rows_sorted.push_back(r);
  }

  std::sort(table.rows_sorted.begin(), table.rows_sorted.end(),
            [](const VerboseRow& x, const VerboseRow& y){ return x.counts > y.counts; });

  const int num_rows = std::min<int>(n_rows_to_display, static_cast<int>(table.rows_sorted.size()));
  table.omitted_rows = table.rows_sorted.size() - num_rows;
  table.rows_sorted.resize(num_rows);

  return table;
}

template <search::concepts::Traits Traits>
void VerboseData<Traits>::to_terminal_text(std::ostream& ss, int n_rows_to_display) const {
  ss << "\nCPU pos eval:\n";

  eigen_util::PrintArrayFormatMap fmt_map{
    {"Player", [&](core::seat_index_t x) { return IO::player_to_str(x); }},
    {"action", [&](float x) { return IO::action_to_str(static_cast<int>(x), mcts_results.action_mode); }},
  };

  Game::GameResults::print_array(mcts_results.value_prior, mcts_results.win_rates, &fmt_map);

  const auto table = build_table_(n_rows_to_display);

  if (Game::Rules::is_chance_mode(table.action_mode)) {
    ss << "******************************\n";
    return;
  }

  static constexpr const char* kCols[] = {"Action", "Prior", "Posterior", "Counts", "Modified"};
  for (std::string_view c : kCols) {
    ss << std::right << std::setw(10) << c;
  }
  ss << "\n";

  for (const auto& r : table.rows_sorted) {
    ss << std::setw(10) << IO::action_to_str(r.action, table.action_mode)
       << std::setw(10) << r.prior
       << std::setw(10) << r.posterior
       << std::setw(10) << r.counts
       << std::setw(10) << r.modified << "\n";
  }

  if (table.omitted_rows > 0) {
    ss << "... " << table.omitted_rows
       << (table.omitted_rows == 1 ? " row not displayed\n" : " rows not displayed\n");
  } else {
    for (int i = 0; i < std::max(0, n_rows_to_display - static_cast<int>(table.rows_sorted.size()) + 1); ++i)
      ss << "\n";
  }
  ss << "******************************\n";
}


}  // namespace generic::alpha0
