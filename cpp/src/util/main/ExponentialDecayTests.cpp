#include "util/GTestUtil.hpp"
#include "util/Math.hpp"

#include <gtest/gtest.h>

#include <vector>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

TEST(ExponentialDecay, JumpToMatchesIterativeStep) {
  struct TestCase {
    float start;
    float end;
    float half_life;
    int steps;
  };

  std::vector<TestCase> cases = {{5.0f, 1.0f, 0.5f, 10},
                                 {5.0f, 1.0f, 0.5f, 0},
                                 {100.0f, 0.0f, 10.0f, 50},
                                 {1.0f, 1.0f, 5.0f, 10},
                                 {10.0f, 0.0f, 0.1f, 5}};

  for (const auto& tc : cases) {
    math::ExponentialDecay iterative(tc.start, tc.end, tc.half_life);
    math::ExponentialDecay jumping(tc.start, tc.end, tc.half_life);

    for (int i = 0; i < tc.steps; ++i) {
      iterative.step();
    }
    jumping.jump_to(tc.steps);

    EXPECT_NEAR(iterative.value(), jumping.value(), 1e-5)
      << "Failed for half_life=" << tc.half_life << ", steps=" << tc.steps;
  }
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
