#include <util/TorchUtil.hpp>

#include <type_traits>

#include <torch/serialize/archive.h>

#include <util/CppUtil.hpp>
#include <util/Exception.hpp>

namespace torch_util {

inline CatchTensorMallocs::CatchTensorMallocs(
    int& catch_count, const torch::Tensor& tensor, const char* var, const char* file, int line, int ignore_count)
: catch_count_(catch_count)
, tensor_(tensor)
, data_ptr_(tensor.data_ptr())
, var_(var)
, file_(file)
, line_(line)
, ignore_count_(ignore_count)
{}

inline CatchTensorMallocs::~CatchTensorMallocs() noexcept(false) {
  if (data_ptr_ == tensor_.data_ptr()) return;
  catch_count_++;
  if (catch_count_ <= ignore_count_) return;
  throw util::Exception("The data memory address of Tensor %s changed %d time%s since it was snapshotted at %s:%d",
                        var_, catch_count_, catch_count_ > 1 ? "s" : "", file_, line_);
}

template<typename... Ts> shape_t to_shape(Ts&&... ts) {
  auto arr = util::to_std_array<int64_t>(std::forward<Ts>(ts)...);
  shape_t vec(arr.begin(), arr.end());
  return vec;
}

inline shape_t zeros_like(const shape_t& shape) {
  return shape_t(shape.size(), 0);
}

inline void pickle_dump(const torch::Tensor& tensor, const boost::filesystem::path& path) {
  auto pickled = torch::pickle_save(tensor);
  std::ofstream fout(path.c_str(), std::ios::out | std::ios::binary);
  fout.write(pickled.data(), pickled.size());
  fout.close();
}

template<typename... SaveToArgs>
void save(const std::map<std::string, torch::Tensor>& tensor_map, SaveToArgs&&... args) {
  torch::serialize::OutputArchive archive(
      std::make_shared<torch::jit::CompilationUnit>());
  for (auto it : tensor_map) {
    archive.write(it.first, it.second);
  }
  archive.save_to(std::forward<SaveToArgs>(args)...);
}

/*
 * TODO: there might be more efficient ways of doing this. One candidate is to use torch::from_blob() to create a
 * tensor from arr. We should profile and change this implementation if appropriate.
 */
template<typename T, size_t N>
inline void copy_to(torch::Tensor tensor, const std::array<T, N>& arr) {
  for (int i = 0; i < int(N); ++i) {
    tensor.index_put_({i}, arr[i]);
  }
}

template<typename T, std::ptrdiff_t... P>
inline void copy_to(
    torch::Tensor to_tensor,
    const Eigen::TensorFixedSize<T, Eigen::Sizes<P...>>& from_tensor)
{
  throw std::exception();
//  from_tensor.slice()
}

inline void init_tensor(torch::Tensor& tensor) {
  tensor = torch::zeros(1);
}

}  // namespace torch_util
