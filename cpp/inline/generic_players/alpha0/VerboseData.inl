#include "generic_players/alpha0/VerboseData.hpp"

namespace generic::alpha0 {

template <search::concepts::Traits Traits>
typename VerboseData<Traits>::Common VerboseData<Traits>::build_common_(
  int n_rows_to_display) const {
  const auto& valid_actions = mcts_results.valid_actions;
  const auto& mcts_counts   = mcts_results.counts;
  const auto& net_policy    = mcts_results.policy_prior;
  const auto& win_rates     = mcts_results.win_rates;
  const auto& net_value     = mcts_results.value_prior;
  const auto  mode          = mcts_results.action_mode;

  Common out;
  out.action_mode = mode;

  // flatten net_value and win_rates (assumed Eigen arrays) to std::vector<float>
  out.net_value_v.assign(net_value.data(), net_value.data() + net_value.size());
  out.win_rates_v.assign(win_rates.data(), win_rates.data() + win_rates.size());

  if (Game::Rules::is_chance_mode(mode)) {
    out.rows_sorted.clear();
    out.omitted_rows = 0;
    return out;
  }

  int num_valid = valid_actions.count();
  out.rows_sorted.reserve(num_valid);

  float total_count = 0.0f;
  for (int a : valid_actions.on_indices()) total_count += mcts_counts(a);
  // avoid div-by-zero
  const float denom = total_count > 0.0f ? total_count : 1.0f;

  for (int a : valid_actions.on_indices()) {
    VerboseRow r;
    r.action    = a;
    r.prior     = net_policy(a);
    r.posterior = mcts_counts(a) / denom;
    r.counts    = mcts_counts(a);
    r.modified  = action_policy(a); // existing member/function
    out.rows_sorted.push_back(r);
  }

  // sort by counts desc
  std::sort(out.rows_sorted.begin(), out.rows_sorted.end(),
            [](const VerboseRow& x, const VerboseRow& y){ return x.counts > y.counts; });

  // clip to N
  const int num_rows = std::min<int>(n_rows_to_display, static_cast<int>(out.rows_sorted.size()));
  out.omitted_rows = static_cast<int>(out.rows_sorted.size()) - num_rows;
  out.rows_sorted.resize(num_rows);

  return out;
}

template <search::concepts::Traits Traits>
void VerboseData<Traits>::to_terminal_text(std::ostream& ss, int n_rows_to_display) const {
  ss << "\nCPU pos eval:\n";

  eigen_util::PrintArrayFormatMap fmt_map{
    {"Player", [&](core::seat_index_t x) { return IO::player_to_str(x); }},
    {"action", [&](float x) { return IO::action_to_str(static_cast<int>(x), mcts_results.action_mode); }},
  };

  // keep existing summary line
  Game::GameResults::print_array(mcts_results.value_prior, mcts_results.win_rates, &fmt_map);

  const auto common = build_common_(n_rows_to_display);

  if (Game::Rules::is_chance_mode(common.action_mode)) {
    ss << "******************************\n";
    return;
  }

  static constexpr const char* kCols[] = {"Action", "Prior", "Posterior", "Counts", "Modified"};
  for (std::string_view c : kCols) {
    ss << std::left << std::setw(12) << c;
  }
  ss << '\n' << std::string(12 * 5, '-') << '\n';

  // print rows
  for (const auto& r : common.rows_sorted) {
    ss << std::setw(12) << IO::action_to_str(r.action, common.action_mode)
       << std::setw(12) << r.prior
       << std::setw(12) << r.posterior
       << std::setw(12) << r.counts
       << std::setw(12) << r.modified << "\n";
  }

  if (common.omitted_rows > 0) {
    ss << "... " << common.omitted_rows
       << (common.omitted_rows == 1 ? " row not displayed\n" : " rows not displayed\n");
  } else {
    // pad to original height (+1 like before)
    for (int i = 0; i < std::max(0, n_rows_to_display - static_cast<int>(common.rows_sorted.size()) + 1); ++i)
      ss << "\n";
  }
  ss << "******************************\n";
}


}  // namespace generic::alpha0
