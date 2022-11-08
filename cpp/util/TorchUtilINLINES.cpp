#include <torch/serialize/archive.h>
#include <torch/serialize/tensor.h>

#include <util/TorchUtil.hpp>

namespace torch_util {

inline shape_t to_shape() { return {}; }

template<typename T, typename... Ts> shape_t to_shape(const std::initializer_list<T>& t, Ts&&... ts) {
  shape_t vec1;
  for (auto x : t) {
    vec1.push_back(x);
  }
  shape_t vec2 = to_shape(std::forward<Ts>(ts)...);
  vec1.insert(vec1.end(), vec2.begin(), vec2.end());
  return vec1;
}

template<typename T, typename... Ts> shape_t to_shape(T t, Ts&&... ts) {
  shape_t vec1 = {int64_t(t)};
  shape_t vec2 = to_shape(std::forward<Ts>(ts)...);
  vec1.insert(vec1.end(), vec2.begin(), vec2.end());
  return vec1;
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
template<typename T>
inline void copy_to(torch::Tensor tensor, const T* arr, int n) {
  for (int i = 0; i < n; ++i) {
    tensor.index_put_({i}, arr[i]);
  }
}

}  // namespace torch_util
