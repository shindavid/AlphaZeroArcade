#include "util/TensorRtUtil.hpp"

#include "util/CudaUtil.hpp"
#include "util/Exceptions.hpp"

#include <magic_enum/magic_enum.hpp>
#include <magic_enum/magic_enum_format.hpp>

#include <NvInfer.h>
#include <cstdint>
#include <cstring>
#include <format>
#include <string>

namespace trt_util {

inline Precision parse_precision(const char* c) {
  if (strcmp(c, "FP32") == 0) return Precision::kFP32;
  if (strcmp(c, "FP16") == 0) return Precision::kFP16;
  if (strcmp(c, "INT8") == 0) return Precision::kINT8;
  throw util::CleanException("Invalid precision '{}'. Valid values are: FP32, FP16, INT8", c);
}

inline const char* precision_to_string(Precision precision) {
  switch (precision) {
    case Precision::kFP32:
      return "FP32";
    case Precision::kFP16:
      return "FP16";
    case Precision::kINT8:
      return "INT8";
    default:
      throw util::Exception("Invalid precision '{}'", precision);
  }
}

inline nvinfer1::BuilderFlag precision_to_builder_flag(Precision precision) {
  switch (precision) {
    case Precision::kFP32:
      return nvinfer1::BuilderFlag::kTF32;
    case Precision::kFP16:
      return nvinfer1::BuilderFlag::kFP16;
    case Precision::kINT8:
      return nvinfer1::BuilderFlag::kINT8;
    default:
      throw util::Exception("Invalid precision '{}'", precision);
  }
}

inline const char* get_version_tag() {
  static std::string tag;
  if (tag.empty()) {
    int maj = getInferLibMajorVersion();
    int min = getInferLibMinorVersion();
    int pat = getInferLibPatchVersion();
    tag = std::to_string(maj) + "." + std::to_string(min) + "." + std::to_string(pat);
  }
  return tag.c_str();
}

inline boost::filesystem::path get_engine_plan_cache_path(
  const std::string& model_architecture_signature, Precision precision,
  uint64_t workspace_size_in_bytes, int batch_size) {
  return std::format(
    "/workspace/mount/TensorRT-cache/v{}/sm_{}/trt_{}/fp_{}/ws_{}/batch_{}/{}.plan", kCacheVersion,
    cuda_util::get_sm_tag(), get_version_tag(), precision_to_string(precision),
    workspace_size_in_bytes, batch_size, model_architecture_signature);
}

}  // namespace trt_util
