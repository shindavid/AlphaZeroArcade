#include "core/WinShareEncoding.hpp"

#include <iostream>

namespace core {

template <int N>
typename WinShareEncoding<N>::Tensor WinShareEncoding<N>::encode(
  const std::array<PlayerResult, N>& outcome) {
  ValueArray shares;
  for (int i = 0; i < N; i++) {
    shares(i) = outcome[i].share;
  }
  return eigen_util::reinterpret_as_tensor(shares);
}

template <int N>
void WinShareEncoding<N>::left_rotate(Tensor& t, core::seat_index_t s) {
  ValueArray& v = eigen_util::reinterpret_as_array(t);
  eigen_util::left_rotate(v, s);
}

template <int N>
void WinShareEncoding<N>::right_rotate(Tensor& t, core::seat_index_t s) {
  ValueArray& v = eigen_util::reinterpret_as_array(t);
  eigen_util::right_rotate(v, s);
}

template <int N>
auto WinShareEncoding<N>::get_data_matrix(const Tensor& net_value, const ValueArray& win_rates) {
  ValueArray net_value_array;
  ValueArray player_array;
  for (int i = 0; i < N; i++) {
    player_array(i) = i;
    net_value_array(i) = net_value(i);
  }
  return eigen_util::concatenate_columns(player_array, net_value_array, win_rates);
}

template <int N>
const std::vector<std::string>& WinShareEncoding<N>::get_column_names() {
  static const std::vector<std::string> columns = {"Player", "Net(W)", "win-rate"};
  return columns;
}

template <int N>
template <typename GameIO>
void WinShareEncoding<N>::print_array(GameIO, const Tensor& net_value,
                                      const ValueArray& win_rates) {
  auto data = get_data_matrix(net_value, win_rates);
  const auto& columns = get_column_names();

  eigen_util::PrintArrayFormatMap fmt_map{
    {"Player", [&](float x, int) { return GameIO::player_to_str(x); }},
  };

  eigen_util::print_array(std::cout, data, columns, &fmt_map);
}

template <int N>
template <typename GameIO>
boost::json::object WinShareEncoding<N>::to_json(GameIO, const Tensor& net_value,
                                                 const ValueArray& win_rates) {
  auto data = get_data_matrix(net_value, win_rates);
  const auto& columns = get_column_names();

  eigen_util::PrintArrayFormatMap fmt_map{
    {"Player", [&](float x, int) { return GameIO::player_to_str(x); }},
  };

  return eigen_util::output_to_json(data, columns, &fmt_map);
}

}  // namespace core
