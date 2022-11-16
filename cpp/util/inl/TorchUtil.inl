#include <util/TorchUtil.hpp>

#include <type_traits>

#include <torch/serialize/archive.h>

#include <util/CppUtil.hpp>

namespace torch_util {

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

}  // namespace torch_util
