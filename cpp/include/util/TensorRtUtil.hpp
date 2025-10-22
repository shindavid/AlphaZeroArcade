#pragma once

#include <boost/filesystem.hpp>

#include <NvInfer.h>
#include <cstdint>
#include <string>

namespace trt_util {

// Increment this whenever you want to invalidate all existing cached engine plans.
constexpr int kCacheVersion = 3;

enum class Precision : uint8_t { kFP32, kFP16, kINT8 };

Precision parse_precision(const char* c);
const char* precision_to_string(Precision precision);
nvinfer1::BuilderFlag precision_to_builder_flag(Precision precision);

const char* get_version_tag();  // "10.11.0"

boost::filesystem::path get_engine_plan_cache_path(const std::string& model_architecture_signature,
                                                   Precision precision,
                                                   uint64_t workspace_size_in_bytes,
                                                   int batch_size);

}  // namespace trt_util

#include "inline/util/TensorRtUtil.inl"
