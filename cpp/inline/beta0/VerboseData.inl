#include "beta0/VerboseData.hpp"

#include "util/EigenUtil.hpp"

#include <iostream>

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
auto VerboseData<EvalSpec>::build_action_data() const {
  const auto& valid_actions = mcts_results.valid_actions;

  int num_valid = valid_actions.count();
  // Zero() calls: not necessary, but silences gcc warning, and is cheap enough
  LocalPolicyArray actions_arr = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray N = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray R = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray V = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray U = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray P = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray Q = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray W = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray pi = LocalPolicyArray::Zero(num_valid);

  int r = 0;
  for (int a : valid_actions.on_indices()) {
    actions_arr(r) = a;
    N(r) = mcts_results.N(a);
    R(r) = mcts_results.RN(a);
    V(r) = mcts_results.AV(a, mcts_results.seat);
    U(r) = mcts_results.AU(a, mcts_results.seat);
    P(r) = mcts_results.P(a);
    Q(r) = mcts_results.AQ(a, mcts_results.seat);
    W(r) = mcts_results.AW(a, mcts_results.seat);
    pi(r) = action_policy(a);
    r++;
  }

  auto data = eigen_util::sort_rows(
    eigen_util::concatenate_columns(actions_arr, N, R, V, U, P, Q, W, pi), 8, false);
  return data;
}

static std::vector<std::string>& get_column_names() {
  static std::vector<std::string> columns = {"action", "N", "R", "V", "U", "P", "Q", "W", "pi"};
  return columns;
}

template <core::concepts::EvalSpec EvalSpec>
boost::json::object VerboseData<EvalSpec>::to_json() const {
  const auto& win_rates = mcts_results.Q;
  const auto& net_value = mcts_results.R;
  core::action_mode_t action_mode = mcts_results.action_mode;

  eigen_util::PrintArrayFormatMap fmt_map{
    {"Player", [&](core::seat_index_t x, int) { return std::string(1, Game::IO::kSeatChars[x]); }},
    {"action", [&](float x, int) { return IO::action_to_str(x, action_mode); }},
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

template <core::concepts::EvalSpec EvalSpec>
void VerboseData<EvalSpec>::to_terminal_text() const {
  std::cout << std::endl << "CPU pos eval:" << std::endl;
  const auto& valid_actions = mcts_results.valid_actions;
  const auto& win_rates = mcts_results.Q;
  const auto& net_value = mcts_results.R;
  core::action_mode_t action_mode = mcts_results.action_mode;

  eigen_util::PrintArrayFormatMap fmt_map{
    {"Player", [&](float x, int) { return IO::player_to_str(x); }},
    {"action", [&](float x, int) { return IO::action_to_str(x, action_mode); }},
  };

  Game::GameResults::print_array(net_value, win_rates, &fmt_map);

  if (Game::Rules::is_chance_mode(mcts_results.action_mode)) return;

  int num_valid = valid_actions.count();
  int num_rows = std::min(num_valid, n_rows_to_display_);
  auto data = build_action_data();
  const auto& columns = get_column_names();

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

}  // namespace beta0
