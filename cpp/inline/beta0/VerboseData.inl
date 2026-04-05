#include "beta0/VerboseData.hpp"

#include "util/EigenUtil.hpp"

#include <iostream>

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
auto VerboseData<EvalSpec>::build_action_data(ActionPrinter& printer) const {
  const auto& valid_moves = mcts_results.valid_moves;

  int num_valid = valid_moves.size();
  // Zero() calls: not necessary, but silences gcc warning, and is cheap enough
  LocalPolicyArray N = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray R = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray V = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray U = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray P = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray Q = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray W = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray pi = LocalPolicyArray::Zero(num_valid);

  int r = 0;
  for (Move move : valid_moves) {
    auto index = PolicyEncoding::to_index(move);
    auto index_s = eigen_util::extend_index(index, mcts_results.seat);

    N(r) = mcts_results.N.coeff(index);
    R(r) = mcts_results.RN.coeff(index);
    V(r) = mcts_results.AV.coeff(index_s);
    U(r) = mcts_results.AU.coeff(index_s);
    P(r) = mcts_results.P.coeff(index);
    Q(r) = mcts_results.AQ.coeff(index_s);
    W(r) = mcts_results.AW.coeff(index_s);
    pi(r) = action_policy.coeff(index);
    r++;
  }

  LocalPolicyArray actions_arr = printer.flat_array();
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
  const auto& valid_moves = mcts_results.valid_moves;
  const auto& win_rates = mcts_results.Q;
  const auto& net_value = mcts_results.R;
  core::game_phase_t game_phase = mcts_results.game_phase;

  boost::json::object cpu_eval = GameResultEncoding::to_json(net_value, win_rates);

  boost::json::object obj;
  obj["cpu_pos_eval"] = std::move(cpu_eval);

  if (Game::Rules::is_chance_phase(game_phase)) return obj;

  ActionPrinter printer(valid_moves);
  auto data = build_action_data(printer);
  const auto& columns = get_column_names();

  eigen_util::PrintArrayFormatMap fmt_map;
  printer.update_format_map(fmt_map);

  obj["actions"] = eigen_util::output_to_json(data, columns, &fmt_map);
  obj["format_funcs"] = boost::json::object{{"Player", "seatToHtml"}};
  return obj;
}

template <core::concepts::EvalSpec EvalSpec>
void VerboseData<EvalSpec>::to_terminal_text() const {
  std::cout << std::endl << "CPU pos eval:" << std::endl;
  const auto& valid_moves = mcts_results.valid_moves;
  const auto& win_rates = mcts_results.Q;
  const auto& net_value = mcts_results.R;
  core::game_phase_t game_phase = mcts_results.game_phase;

  GameResultEncoding::print_array(net_value, win_rates);

  if (Game::Rules::is_chance_phase(game_phase)) return;

  ActionPrinter printer(valid_moves);

  eigen_util::PrintArrayFormatMap fmt_map;
  printer.update_format_map(fmt_map);

  int num_valid = valid_moves.size();
  int num_rows = std::min(num_valid, n_rows_to_display_);
  auto data = build_action_data(printer);
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
