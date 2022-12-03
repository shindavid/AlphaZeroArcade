#pragma once

/*
 * Various util functions that make the torch library more pleasant to use.
 */
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <torch/torch.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include <util/CppUtil.hpp>

/*
 * Some torch::Tensor API's lead to dynamic memory allocation under the hood. It is desirable to avoid this wherever
 * possible, for performance reasons. Unfortunately, the presence of dynamic memory allocations is often not
 * obvious when looking at the caller code.
 *
 * To illustrate:
 *
 * t = t.resize({5});  // no dynamic memory allocation
 * t = t.softmax(0);  // dynamic memory allocation!
 *
 * The proper way to do the above softmax operation "in-place" is:
 *
 * torch::softmax_out(t, t, 0);  // no dynamic memory allocation
 *
 * The CATCH_TENSOR_MALLOCS macro is a tool that helps to automatically detect dynamic memory allocations.
 *
 * Usage:
 *
 * void func() {
 *   ...
 *   CATCH_TENSOR_MALLOCS(tensor);
 *   // do stuff with tensor
 *   ...
 * }
 *
 * In release builds, the CATCH_TENSOR_MALLOCS line is simply compiled out.
 *
 * In debug builds, the line becomes something like:
 *
 * torch_util::CatchTensorMallocs __unique_var_name_12345(tensor, "tensor", __FILE__, __LINE__);
 *
 * This constructs a CatchTensorMallocs object, passing tensor as an argument. The memory address of the tensor's data
 * is recorded in the constructor. When the CatchTensorMallocs destructor is called, the memory address is checked
 * again. If the address has changed, an exception is thrown.
 *
 * CATCH_TENSOR_MALLOCS() can take an optional second int argument, N. If this is passed, then the first N exceptions
 * thrown by that exact CATCH_TENSOR_MALLOC (identified by file/line) are effectively caught and ignored. The default
 * value of this argument is 1. The rationale for this default is that we care most about preventing dynamic memory
 * allocations in functions that are called repeatedly, but in those contexts, we want the convenience of lazily
 * assigning data to the Tensor rather than having to initialize its shape/dtype up-front.
 */
#ifdef NDEBUG
#define CATCH_TENSOR_MALLOCS(...)
#else  // NDEBUG
#define CATCH_TENSOR_MALLOCS(t, ...) \
static int CONCAT(__unique_var1_, __LINE__) = 0; \
torch_util::CatchTensorMallocs CONCAT(__unique_var2_, __LINE__) \
    (CONCAT(__unique_var1_, __LINE__), t, #t, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__);
#endif  // NDEBUG

namespace torch_util {

/*
 * See documentation for macro CATCH_TENSOR_MALLOCS().
 */
class CatchTensorMallocs {
public:
  CatchTensorMallocs(int& catch_count, const torch::Tensor& tensor, const char* var, const char* file, int line,
                     int ignore_count=1);
  ~CatchTensorMallocs() noexcept(false);

private:
  int& catch_count_;
  const torch::Tensor& tensor_;
  const void* data_ptr_;
  const char* var_;
  const char* file_;
  const int line_;
  const int ignore_count_;
};

using shape_t = std::vector<int64_t>;

/*
 * Smash together integral and std::array arguments into a single shape_t. Without this helper
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
 * A default-constructed torch::Tensor cannot be used. This function assigns
 * the tensor to an arbitrary value, so that the tensor can be re-assigned
 * later.
 */
void init_tensor(torch::Tensor& tensor);

}  // namespace torch_util

#include <util/inl/TorchUtil.inl>
