#include <util/HighFiveUtil.hpp>

namespace hi5 {

template<> shape_t to_shape() { return {}; }

template<typename... Ts> shape_t to_shape(Ts&&... ts, size_t s) {
  shape_t shape = to_shape(std::forward<Ts>(ts)...);
  shape.push_back(s);
  return shape;
}

template<typename... Ts> shape_t to_shape(Ts&&... ts, const std::initializer_list<size_t>& s) {
  shape_t shape = to_shape(std::forward<Ts>(ts)...);
  shape.insert(shape.end(), s);
  return shape;
}

inline shape_t zeros_like(const shape_t& shape) {
  shape_t zeros;
  for (auto x : shape) {
    zeros.push_back(0);
  }
  return zeros;
}

}
