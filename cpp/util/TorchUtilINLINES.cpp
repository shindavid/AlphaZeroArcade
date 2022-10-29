#include <torch/serialize/archive.h>
#include <torch/serialize/tensor.h>

#include <util/TorchUtil.hpp>

namespace torch_util {

namespace detail {

template<> int_vec_t to_shape_helper() { return {}; }

template<typename... Ts> int_vec_t to_shape_helper(Ts&&... ts, int64_t s) {
  int_vec_t shape = to_shape_helper(std::forward<Ts>(ts)...);
  shape.push_back(s);
  return shape;
}

template<typename... Ts> int_vec_t to_shape_helper(Ts&&... ts, const std::initializer_list<int64_t>& s) {
  int_vec_t shape = to_shape_helper(std::forward<Ts>(ts)...);
  shape.insert(shape.end(), s);
  return shape;
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

template<typename T>
inline void copy_to(torch::Tensor tensor, const T* arr, int n) {
  for (int i = 0; i < n; ++i) {
    tensor.index_put_({i}, arr[i]);
  }
}

}  // namespace torch_util
