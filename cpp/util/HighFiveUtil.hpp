#pragma once

/*
 * Some util functions that make the HighFive library more pleasant to use.
 */

#include <cstdint>
#include <vector>

namespace hi5 {

using shape_t = std::vector<size_t>;

/*
 * Smash together size_t and std::initializer_list<size_t> arguments into a single shape_t.
 */
template<typename... Ts> shape_t to_shape(Ts&&... ts);

shape_t zeros_like(const shape_t& shape);

}  // namespace hi5

#include <util/HighFiveUtilINLINES.cpp>
