#include "generic_players/alpha0/VerboseData.hpp"

namespace generic::alpha0 {

template <search::concepts::Traits Traits>
void VerboseData<Traits>::build_table() const {
  table_.clear();

  const auto& valid_actions = mcts_results.valid_actions;
  const auto& mcts_counts = mcts_results.counts;
  const auto& net_policy = mcts_results.policy_prior;
  const auto& win_rates = mcts_results.win_rates;
  const auto& net_value = mcts_results.value_prior;
  const auto action_mode = mcts_results.action_mode;

  table_.action_mode = action_mode;

  table_.net_value_v.assign(net_value.data(), net_value.data() + net_value.size());
  table_.win_rates_v.assign(win_rates.data(), win_rates.data() + win_rates.size());

  if (Game::Rules::is_chance_mode(action_mode)) {
    table_.rows_sorted.clear();
  }

  int num_valid = valid_actions.count();
  table_.rows_sorted.reserve(num_valid);

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
    table_.rows_sorted.push_back(r);
  }

  std::sort(table_.rows_sorted.begin(), table_.rows_sorted.end(),
            [](const VerboseRow& x, const VerboseRow& y){ return x.counts > y.counts; });

}

template <search::concepts::Traits Traits>
void VerboseData<Traits>::to_terminal_text(std::ostream& ss, int n_rows_to_display) const {
  ss << "\nCPU pos eval:\n";

  eigen_util::PrintArrayFormatMap fmt_map{
    {"Player", [&](core::seat_index_t x) { return IO::player_to_str(x); }},
    {"action", [&](float x) { return IO::action_to_str(static_cast<int>(x), mcts_results.action_mode); }},
  };

  Game::GameResults::print_array(mcts_results.value_prior, mcts_results.win_rates, &fmt_map);

  const int num_rows = std::min<int>(n_rows_to_display, static_cast<int>(table_.rows_sorted.size()));
  int omitted_rows = table_.rows_sorted.size() - num_rows;

  if (Game::Rules::is_chance_mode(table_.action_mode)) {
    ss << "******************************\n";
    return;
  }

  static constexpr const char* kCols[] = {"Action", "Prior", "Posterior", "Counts", "Modified"};
  for (std::string_view c : kCols) {
    ss << std::right << std::setw(10) << c;
  }
  ss << "\n";

  for (int i = 0; i < num_rows; ++i) {
    const VerboseRow& r = table_.rows_sorted[i];
    ss << std::setw(10) << IO::action_to_str(r.action, table_.action_mode)
       << std::setw(10) << r.prior
       << std::setw(10) << r.posterior
       << std::setw(10) << r.counts
       << std::setw(10) << r.modified << "\n";
  }

  if (omitted_rows > 0) {
    ss << "... " << omitted_rows
       << (omitted_rows == 1 ? " row not displayed\n" : " rows not displayed\n");
  } else {
    for (int i = 0; i < std::max(0, n_rows_to_display - static_cast<int>(table_.rows_sorted.size()) + 1); ++i)
      ss << "\n";
  }
  ss << "******************************\n";
}

template <search::concepts::Traits Traits>
boost::json::object VerboseData<Traits>::to_json() const {
  boost::json::object obj;
  obj["action_mode"] = static_cast<int>(table_.action_mode);

  // cpu pos eval
  boost::json::object eval;
  eval["net_value"] = boost::json::array(table_.net_value_v.begin(), table_.net_value_v.end());
  eval["win_rates"] = boost::json::array(table_.win_rates_v.begin(), table_.win_rates_v.end());
  obj["cpu_pos_eval"] = std::move(eval);

  // actions
  boost::json::array rows;
  rows.reserve(table_.rows_sorted.size());
  for (const auto& r : table_.rows_sorted) {
    boost::json::object row;
    row["action_id"] = r.action;
    row["action_str"] = IO::action_to_str(r.action, table_.action_mode);
    row["prior"] = r.prior;
    row["posterior"] = r.posterior;
    row["counts"] = r.counts;
    row["modified"] = r.modified;
    rows.push_back(std::move(row));
  }
  obj["actions"] = std::move(rows);

  return obj;
}

template <search::concepts::Traits Traits>
void VerboseData<Traits>::set(const PolicyTensor& policy, const SearchResults& results) {
  action_policy = policy;
  mcts_results = results;
  build_table();
  initialized = true;
}

}  // namespace generic::alpha0
