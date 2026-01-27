#include "beta0/VerboseData.hpp"

#include "util/EigenUtil.hpp"

namespace beta0 {

template <core::concepts::Game Game>
auto VerboseData<Game>::build_action_data() const {
  const auto& valid_actions = mcts_results.valid_actions;
  const auto& net_policy = mcts_results.P;

  int num_valid = valid_actions.count();
  // Zero() calls: not necessary, but silences gcc warning, and is cheap enough
  LocalPolicyArray actions_arr = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray net_policy_arr = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray action_policy_arr = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray V = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray U = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray Q = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray W = LocalPolicyArray::Zero(num_valid);

  int r = 0;
  for (int a : valid_actions.on_indices()) {
    actions_arr(r) = a;
    V(r) = mcts_results.AV(a, mcts_results.seat);
    U(r) = mcts_results.AU(a, mcts_results.seat);
    Q(r) = mcts_results.AQ(a, mcts_results.seat);
    W(r) = mcts_results.AW(a, mcts_results.seat);
    net_policy_arr(r) = net_policy(a);
    action_policy_arr(r) = action_policy(a);
    r++;
  }

  auto data = eigen_util::sort_rows(
    eigen_util::concatenate_columns(actions_arr, V, U, net_policy_arr, Q, W, action_policy_arr), 6, false);
  return data;
}

static std::vector<std::string>& get_column_names() {
  static std::vector<std::string> columns = {"action", "V", "U", "P", "Q", "W", "pi"};
  return columns;
}

template <core::concepts::Game Game>
boost::json::object VerboseData<Game>::to_json() const {

  const auto& win_rates = mcts_results.Q;
  const auto& net_value = mcts_results.R;
  core::action_mode_t action_mode = mcts_results.action_mode;

  eigen_util::PrintArrayFormatMap fmt_map{
    {"Player", [&](core::seat_index_t x) { return std::string(1, Game::IO::kSeatChars[x]); }},
    {"action", [&](float x) { return IO::action_to_str(x, action_mode); }},
  };

  boost::json::object obj;
  boost::json::object cpu_eval = Game::GameResults::to_json(net_value, win_rates, &fmt_map);
  obj["cpu_pos_eval"] = std::move(cpu_eval);

  if (Game::Rules::is_chance_mode(action_mode)) return obj;

  auto data = build_action_data();
  const auto& columns = get_column_names();
  obj["actions"] = eigen_util::output_to_json(data, columns, &fmt_map);
  obj["format_funcs"] = boost::json::object{{"Player", "seatToHtml"}};
  return obj;
}

}  // namespace beta0
