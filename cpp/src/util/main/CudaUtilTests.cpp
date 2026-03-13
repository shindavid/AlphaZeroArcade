#include "util/CudaUtil.hpp"
#include "util/GTestUtil.hpp"

#include <gtest/gtest.h>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

TEST(cuda_util, cuda_device_to_ordinal) {
  EXPECT_EQ(cuda_util::cuda_device_to_ordinal("cuda:0"), 0);
  EXPECT_EQ(cuda_util::cuda_device_to_ordinal("0"), 0);
  EXPECT_EQ(cuda_util::cuda_device_to_ordinal("cuda:1"), 1);
  EXPECT_EQ(cuda_util::cuda_device_to_ordinal("1"), 1);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
