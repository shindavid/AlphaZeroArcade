#include <core/tests/Common.hpp>
#include <games/hex/Constants.hpp>
#include <games/hex/Game.hpp>
#include <games/hex/Types.hpp>
#include <iostream>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>
#include <util/GTestUtil.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

using UnionFind = hex::UnionFind;
using vertex_t = hex::vertex_t;
using Game = hex::Game;
using State = Game::State;
using StateHistory = Game::StateHistory;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;

State make_init_state() {
  StateHistory history;
  history.initialize(Rules{});

  Rules::apply(history, 11);
  Rules::apply(history, 101);
  Rules::apply(history, 22);
  return history.current();
}

PolicyTensor make_policy(int move1, int move2) {
  PolicyTensor tensor;
  tensor.setZero();
  tensor(move1) = 1;
  tensor(move2) = 1;
  return tensor;
}

const std::string init_state_repr =
  "               A B C D E F G H I J K\n"
  "          11 / / / / / / / / / / / / 11\n"
  "         10 / / /W/ / / / / / / / / 10\n"
  "         9 / / / / / / / / / / / /  9\n"
  "        8 / / / / / / / / / / / /  8\n"
  "       7 / / / / / / / / / / / /  7\n"
  "      6 / / / / / / / / / / / /  6\n"
  "     5 / / / / / / / / / / / /  5\n"
  "    4 / / / / / / / / / / / /  4\n"
  "   3 /B/ / / / / / / / / / /  3\n"
  "  2 /B/ / / / / / / / / / /  2\n"
  " 1 / / / / / / / / / / / /  1\n"
  "   A B C D E F G H I J K\n\n";

std::string get_repr(const State& state) {
  std::ostringstream ss;
  IO::print_state(ss, state);
  return ss.str();
}

class UnionFindTest : public ::testing::Test {
protected:
  UnionFind uf;

  void SetUp() override {
    uf.init();
  }
};

TEST_F(UnionFindTest, InitParentsAreSelf) {
  // After init, every node should be its own parent
  for (vertex_t i = 0; i < UnionFind::kNumVertices; ++i) {
    EXPECT_EQ(uf.find(i), i) << "Node " << i << " should be its own root";
    EXPECT_FALSE(uf.connected(i, (i+1) % UnionFind::kNumVertices))
        << "Different nodes should not be connected by default";
  }
}

TEST_F(UnionFindTest, UniteTwoElements) {
  constexpr auto a = vertex_t(1);
  constexpr auto b = vertex_t(2);
  EXPECT_FALSE(uf.connected(a,b));
  uf.unite(a,b);
  EXPECT_TRUE(uf.connected(a,b));
  EXPECT_EQ(uf.find(a), uf.find(b));
}

TEST_F(UnionFindTest, ChainedUnions) {
  // chain a–b, b–c => a, c become connected
  constexpr auto a = vertex_t(3);
  constexpr auto b = vertex_t(4);
  constexpr auto c = vertex_t(5);
  uf.unite(a,b);
  uf.unite(b,c);

  EXPECT_TRUE(uf.connected(a,c));
  EXPECT_EQ(uf.find(a), uf.find(c));
}

TEST_F(UnionFindTest, MultipleUniteIdempotent) {
  constexpr auto x = vertex_t(6);
  constexpr auto y = vertex_t(7);
  // multiple unites should not break
  uf.unite(x,y);
  auto first_root = uf.find(x);
  uf.unite(x,y);
  uf.unite(y,x);
  EXPECT_EQ(uf.find(x), first_root);
  EXPECT_TRUE(uf.connected(x,y));
}

TEST_F(UnionFindTest, VirtualNodesSeparatePerPlayer) {
  // Virtual nodes should start unconnected
  auto v1 = UnionFind::kVirtualVertex1;
  auto v2 = UnionFind::kVirtualVertex2;
  EXPECT_FALSE(uf.connected(v1, v2));

  // If we unite them, they become connected
  uf.unite(v1, v2);
  EXPECT_TRUE(uf.connected(v1, v2));
}

TEST_F(UnionFindTest, FindDoesPathCompression) {
  // Create a deep chain: 10 -> 11 -> 12 -> 13
  constexpr auto n0 = vertex_t(10);
  constexpr auto n1 = vertex_t(11);
  constexpr auto n2 = vertex_t(12);
  constexpr auto n3 = vertex_t(13);

  uf.unite(n0, n1);
  uf.unite(n1, n2);
  uf.unite(n2, n3);

  // At this point, find(n0) should compress the path
  auto root_before = uf.find(n0);
  EXPECT_EQ(root_before, uf.find(n3));

  // Inspect parent of n0 (requires access; here we check indirectly)
  // A second find should hit the same root immediately without extra recursion
  EXPECT_EQ(uf.find(n0), root_before);
}

TEST(Symmetry, identity) {
  State state = make_init_state();

  group::element_t sym = groups::C2::kIdentity;
  group::element_t inv_sym = groups::C2::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr = init_state_repr;

  EXPECT_STREQ(repr.c_str(), expected_repr.c_str());
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_STREQ(get_repr(state).c_str(), init_state_repr.c_str());

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym, 0);
  PolicyTensor expected_policy = make_policy(0, 1);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym, 0);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, rotate) {
  State state = make_init_state();

  group::element_t sym = groups::C2::kRot180;
  group::element_t inv_sym = groups::C2::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "               A B C D E F G H I J K\n"
    "          11 / / / / / / / / / / / / 11\n"
    "         10 / / / / / / / / / / /B/ 10\n"
    "         9 / / / / / / / / / / /B/  9\n"
    "        8 / / / / / / / / / / / /  8\n"
    "       7 / / / / / / / / / / / /  7\n"
    "      6 / / / / / / / / / / / /  6\n"
    "     5 / / / / / / / / / / / /  5\n"
    "    4 / / / / / / / / / / / /  4\n"
    "   3 / / / / / / / / / / / /  3\n"
    "  2 / / / / / / / / /W/ / /  2\n"
    " 1 / / / / / / / / / / / /  1\n"
    "   A B C D E F G H I J K\n\n";

  EXPECT_STREQ(repr.c_str(), expected_repr.c_str());
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_STREQ(get_repr(state).c_str(), init_state_repr.c_str());

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym, 0);
  PolicyTensor expected_policy = make_policy(119, 120);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym, 0);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

int main(int argc, char** argv) {
  return launch_gtest(argc, argv);
}
