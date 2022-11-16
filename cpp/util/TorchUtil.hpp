#pragma once

/*
 * Various util functions that make the torch library more pleasnt to use.
 */

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <torch/torch.h>

namespace torch_util {

using shape_t = std::vector<int64_t>;

/*
 * Smash together integral  and std::array arguments into a single shape_t. Without this helper
 * function, constructing shapes through concatenation is cumbersome.
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
 * Our torch_util::save() function is similar, except we get to choose the string keys explicitly.
 */
template<typename... SaveToArgs>
void save(const std::map<std::string, torch::Tensor>& tensor_map, SaveToArgs&&... args);

/*
 * Copy arr[0], arr[1], ..., arr[N-1] to tensor[0], tensor[1], ..., tensor[N-1].
 */
template<typename T, size_t N>
void copy_to(torch::Tensor tensor, const std::array<T, N>& arr);

}  // namespace torch_util

#include <util/inl/TorchUtil.inl>
