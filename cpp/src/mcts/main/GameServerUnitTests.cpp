#include <gtest/gtest.h>

// Dummy test case to check build and run
TEST(BasicTest, SanityCheck) {
    EXPECT_EQ(1 + 1, 2);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
