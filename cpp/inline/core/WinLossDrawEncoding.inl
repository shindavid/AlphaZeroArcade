#include "core/WinLossDrawEncoding.hpp"

#include <algorithm>
#include <iostream>

namespace core {

template <concepts::Game Game>
typename WinLossDrawEncoding<Game>::Tensor WinLossDrawEncoding<Game>::encode(
  const GameOutcome& outcome) {
  Tensor t;
  t.setZero();
  // outcome[0] describes the result from seat-0's perspective
  t(static_cast<int>(outcome[0].kind)) = 1.0f;
  return t;
}

template <concepts::Game Game>
typename WinLossDrawEncoding<Game>::ValueArray WinLossDrawEncoding<Game>::to_value_array(
  const Tensor& t) {
  ValueArray a;
  a(0) = t(0) + 0.5f * t(2);
  a(1) = t(1) + 0.5f * t(2);
  return a;
}

template <concepts::Game Game>
void WinLossDrawEncoding<Game>::left_rotate(Tensor& t, core::seat_index_t s) {
  if (s) {
    std::swap(t(0), t(1));
  }
}

template <concepts::Game Game>
auto WinLossDrawEncoding<Game>::get_data_matrix(const Tensor& net_value,
                                                const ValueArray& win_rates) {
  ValueArray net_value_array;
  ValueArray net_draw_array;
  ValueArray player_array;
  for (int i = 0; i < 2; i++) {
    player_array(i) = i;
    net_value_array(i) = net_value(i);
    net_draw_array(i) = net_value(2);
  }
  return eigen_util::concatenate_columns(player_array, net_value_array, net_draw_array, win_rates);
}

template <concepts::Game Game>
inline const std::vector<std::string>& WinLossDrawEncoding<Game>::get_column_names() {
  static const std::vector<std::string> columns = {"Player", "Net(W)", "Net(D)", "win-rate"};
  return columns;
}

template <concepts::Game Game>
inline void WinLossDrawEncoding<Game>::print_array(const Tensor& net_value,
                                                   const ValueArray& win_rates) {
  auto data = get_data_matrix(net_value, win_rates);
  const auto& columns = get_column_names();

  eigen_util::PrintArrayFormatMap fmt_map{
    {"Player", [&](float x, int) { return GameIO::player_to_str(x); }},
  };

  eigen_util::print_array(std::cout, data, columns, &fmt_map);
}

template <concepts::Game Game>
inline boost::json::object WinLossDrawEncoding<Game>::to_json(const Tensor& net_value,
                                                              const ValueArray& win_rates) {
  auto data = get_data_matrix(net_value, win_rates);
  const auto& columns = get_column_names();

  eigen_util::PrintArrayFormatMap fmt_map{
    {"Player", [&](float x, int) { return GameIO::player_to_str(x); }},
  };

  return eigen_util::output_to_json(data, columns, &fmt_map);
}

}  // namespace core
