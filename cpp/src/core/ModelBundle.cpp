#include "core/ModelBundle.hpp"

#include "util/Exceptions.hpp"

#include <onnx/onnx_pb.h>

#include <cstring>
#include <string_view>

namespace core {

namespace {

constexpr std::string_view kNnuePrefix = "nnue/";

// Extract a FLOAT-typed onnx::TensorProto into a std::vector<float>.
std::vector<float> extract_float_tensor(const onnx::TensorProto& tensor) {
  if (tensor.data_type() != onnx::TensorProto::FLOAT) {
    throw util::Exception(
      "ModelBundle: NNUE initializer '{}' has unsupported dtype {} "
      "(expected FLOAT)",
      tensor.name(), (int)tensor.data_type());
  }

  // Compute element count from dims.
  int64_t n_elements = 1;
  for (int i = 0; i < tensor.dims_size(); ++i) {
    n_elements *= tensor.dims(i);
  }
  if (n_elements < 0) {
    throw util::Exception("ModelBundle: NNUE initializer '{}' has negative element count",
                          tensor.name());
  }

  std::vector<float> result(static_cast<size_t>(n_elements));

  // ONNX TensorProto stores float data either in raw_data (little-endian bytes) or in
  // float_data (repeated float field). Prefer raw_data when present.
  if (tensor.has_raw_data()) {
    const std::string& raw = tensor.raw_data();
    if (raw.size() != n_elements * sizeof(float)) {
      throw util::Exception("ModelBundle: NNUE initializer '{}' raw_data size {} != expected {}",
                            tensor.name(), raw.size(), n_elements * sizeof(float));
    }
    std::memcpy(result.data(), raw.data(), raw.size());
  } else {
    if (tensor.float_data_size() != n_elements) {
      throw util::Exception(
        "ModelBundle: NNUE initializer '{}' float_data size {} != "
        "expected {}",
        tensor.name(), tensor.float_data_size(), n_elements);
    }
    for (int64_t i = 0; i < n_elements; ++i) {
      result[i] = tensor.float_data(i);
    }
  }

  return result;
}

}  // namespace

bool parse_model_bundle(std::shared_ptr<const std::vector<char>> raw, ModelBundle& out) {
  out.onnx_bytes = raw;
  out.nnue_weights.clear();

  if (!raw || raw->empty()) return false;

  onnx::ModelProto model;
  if (!model.ParseFromArray(raw->data(), static_cast<int>(raw->size()))) {
    return false;
  }

  const auto& graph = model.graph();
  for (int i = 0; i < graph.initializer_size(); ++i) {
    const auto& tensor = graph.initializer(i);
    const std::string& name = tensor.name();
    if (name.size() <= kNnuePrefix.size()) continue;
    if (std::string_view(name).substr(0, kNnuePrefix.size()) != kNnuePrefix) continue;

    std::string key = name.substr(kNnuePrefix.size());
    out.nnue_weights.emplace(std::move(key), extract_float_tensor(tensor));
  }

  return true;
}

}  // namespace core
