#include <torch/serialize/archive.h>
#include <torch/serialize/tensor.h>

#include <util/TorchUtil.hpp>

namespace torch_util {

namespace detail {

using int_vec_t = std::vector<int64_t>;

inline int_vec_t to_shape_helper() { return {}; }

template<typename T, typename... Ts> int_vec_t to_shape_helper(const std::initializer_list<T>& t, Ts&&... ts) {
  int_vec_t vec1;
  for (auto x : t) {
    vec1.push_back(x);
  }
  int_vec_t vec2 = to_shape_helper(std::forward<Ts>(ts)...);
  vec1.insert(vec1.end(), vec2.begin(), vec2.end());
  return vec1;
}

template<typename T, typename... Ts> int_vec_t to_shape_helper(T t, Ts&&... ts) {
  int_vec_t vec1 = {int64_t(t)};
  int_vec_t vec2 = to_shape_helper(std::forward<Ts>(ts)...);
  vec1.insert(vec1.end(), vec2.begin(), vec2.end());
  return vec1;
}

}  // namespace detail

template<typename... Ts> shape_t to_shape(Ts&&... ts) {
  return detail::to_shape_helper(std::forward<Ts>(ts)...);
}

inline shape_t zeros_like(const shape_t& shape) {
  return detail::int_vec_t(shape.size(), 0);
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
