#include "core/WinLossEncoding.hpp"

#include <iostream>

namespace core {

template <concepts::Game Game>
typename WinLossEncoding<Game>::Tensor WinLossEncoding<Game>::encode(const GameOutcome& outcome) {
  Tensor t;
  t.setZero();
  t(0) = (outcome[0].kind == PlayerResult::kWin) ? 1.0f : 0.0f;
  t(1) = (outcome[1].kind == PlayerResult::kWin) ? 1.0f : 0.0f;
  return t;
}

template <concepts::Game Game>
auto WinLossEncoding<Game>::get_data_matrix(const Tensor& net_value, const ValueArray& win_rates) {
  ValueArray player_array;
  for (int i = 0; i < 2; i++) {
    player_array(i) = i;
  }
  return eigen_util::concatenate_columns(player_array, win_rates);
}

template <concepts::Game Game>
inline const std::vector<std::string>& WinLossEncoding<Game>::get_column_names() {
  static const std::vector<std::string> columns = {"Player", "win-rate"};
  return columns;
}

template <concepts::Game Game>
inline void WinLossEncoding<Game>::print_array(const Tensor& net_value,
                                               const ValueArray& win_rates) {
  auto data = get_data_matrix(net_value, win_rates);
  const auto& columns = get_column_names();

  eigen_util::PrintArrayFormatMap fmt_map{
    {"Player", [&](float x, int) { return GameIO::player_to_str(x); }},
  };

  eigen_util::print_array(std::cout, data, columns, &fmt_map);
}

template <concepts::Game Game>
inline boost::json::object WinLossEncoding<Game>::to_json(const Tensor& net_value,
                                                          const ValueArray& win_rates) {
  auto data = get_data_matrix(net_value, win_rates);
  const auto& columns = get_column_names();

  eigen_util::PrintArrayFormatMap fmt_map{
    {"Player", [&](float x, int) { return GameIO::player_to_str(x); }},
  };

  return eigen_util::output_to_json(data, columns, &fmt_map);
}

}  // namespace core
