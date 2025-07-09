#include <util/GTestUtil.hpp>
#include <util/mit/mit.hpp>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

class DeterministicTest : public testing::Test {
 public:

};

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
