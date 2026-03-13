#include "util/GTestUtil.hpp"
#include "util/TensorRtUtil.hpp"

#include <gtest/gtest.h>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

TEST(TensorRtUtil, parse_precision) {
  EXPECT_EQ(trt_util::parse_precision("FP16"), trt_util::Precision::kFP16);
  EXPECT_EQ(trt_util::parse_precision("FP32"), trt_util::Precision::kFP32);
  EXPECT_EQ(trt_util::parse_precision("INT8"), trt_util::Precision::kINT8);

  EXPECT_THROW(trt_util::parse_precision("FP64"), util::CleanException);
}

TEST(TensorRtUtil, get_version_tag) {
  const char* version_tag = trt_util::get_version_tag();
  EXPECT_STREQ(version_tag, "10.11.0");
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
