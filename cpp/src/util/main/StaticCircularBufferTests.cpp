#include "util/GTestUtil.hpp"
#include "util/StaticCircularBuffer.hpp"

#include <gtest/gtest.h>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

TEST(StaticCircularBuffer, BasicOperations) {
  util::StaticCircularBuffer<int, 3> buffer;

  EXPECT_TRUE(buffer.empty());
  EXPECT_EQ(buffer.size(), 0);

  buffer.push_back(1);
  buffer.push_back(2);
  buffer.push_back(3);
  EXPECT_EQ(buffer.size(), 3);
  EXPECT_EQ(buffer.back(), 3);

  buffer.push_back(4);
  EXPECT_EQ(buffer.size(), 3);
  EXPECT_EQ(buffer.back(), 4);
}

TEST(StaticCircularBuffer, PushFront) {
  util::StaticCircularBuffer<int, 3> buffer;

  buffer.push_front(1);
  buffer.push_front(2);
  buffer.push_front(3);
  EXPECT_EQ(buffer.size(), 3);
  EXPECT_EQ(buffer.front(), 3);
  EXPECT_EQ(buffer.back(), 1);

  buffer.push_front(4);
  EXPECT_EQ(buffer.size(), 3);
  EXPECT_EQ(buffer.front(), 4);
  EXPECT_EQ(buffer.back(), 2);

  buffer.push_back(5);
  EXPECT_EQ(buffer.size(), 3);
  EXPECT_EQ(buffer.front(), 3);
  EXPECT_EQ(buffer.back(), 5);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
