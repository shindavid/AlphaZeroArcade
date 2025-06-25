#include <core/tests/Common.hpp>
#include <games/hex/Constants.hpp>
#include <games/hex/Game.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>
#include <util/GTestUtil.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

/*
 * Tests hex symmetry classes.
 */

using Game = hex::Game;
using State = Game::State;
using StateHistory = Game::StateHistory;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;

TEST(Dummy, dummy) {
  EXPECT_EQ(1, 1);
}

int main(int argc, char** argv) {
  return launch_gtest(argc, argv);
}
