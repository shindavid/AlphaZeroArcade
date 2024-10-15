#include <core/tests/Common.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
