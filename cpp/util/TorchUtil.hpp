#pragma once

/*
 * Various util functions that make the torch library more pleasnt to use.
 */

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <torch/torch.h>

namespace torch_util {

namespace detail {
using int_vec_t = std::vector<int64_t>;
template<typename... Ts> int_vec_t to_shape_helper(Ts&&... ts);
}  // namespace detail

using shape_t = at::IntArrayRef;

/*
 * Smash together int64_t and std::initializer_list<int64_t> arguments into a single shape_t.
 */
template<typename... Ts> shape_t to_shape(Ts&&... ts);

shape_t zeros_like(const shape_t& shape);

void pickle_dump(const torch::Tensor& tensor, const boost::filesystem::path& path);

/*
 * The torch::save() function takes a vector, vec, of Tensor's as its first argument and writes the following
 * (string, tensor) mappings to disk:
 *
 * "0" -> vec[0]
 * "1" -> vec[1]
 * "2" -> vec[2]
 * ...
 *
 * Our torch_util::save() function is similar, except we get to choose mapping keys explicitly.
 */
template<typename... SaveToArgs>
void save(const std::map<std::string, torch::Tensor>& tensor_map, SaveToArgs&&... args);

/*
 * Copy arr[0], arr[1], ..., arr[n-1] to tensor[0], tensor[1], ..., tensor[n-1].
 */
template<typename T>
void copy_to(torch::Tensor tensor, const T* arr, int n);

}  // namespace torch_util

#include <util/TorchUtilINLINES.cpp>
