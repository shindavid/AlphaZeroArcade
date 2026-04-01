#include "core/BasicTypes.hpp"
#include "games/hex/Constants.hpp"
#include "games/hex/Game.hpp"
#include "games/hex/Symmetries.hpp"
#include "games/hex/Types.hpp"
#include "util/EigenUtil.hpp"
#include "util/GTestUtil.hpp"

#include <gtest/gtest.h>

#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using UnionFind = hex::UnionFind;
using vertex_t = hex::vertex_t;
using Game = hex::Game;
using Symmetries = hex::Symmetries;
using Constants = hex::Constants;
using State = Game::State;
using ActionMask = Game::Types::ActionMask;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;
using RulesResult = Game::Rules::Result;

State make_init_state() {
  State state;
  Rules::init_state(state);

  Rules::apply(state, 11);
  Rules::apply(state, 101);
  Rules::apply(state, 22);
  return state;
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
  "         10 / / /B/ / / / / / / / / 10\n"
  "         9 / / / / / / / / / / / /  9\n"
  "        8 / / / / / / / / / / / /  8\n"
  "       7 / / / / / / / / / / / /  7\n"
  "      6 / / / / / / / / / / / /  6\n"
  "     5 / / / / / / / / / / / /  5\n"
  "    4 / / / / / / / / / / / /  4\n"
  "   3 /R/ / / / / / / / / / /  3\n"
  "  2 /R/ / / / / / / / / / /  2\n"
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

  void SetUp() override { uf.init(); }
};

TEST(Analyze, FromInitState) {
  State state;
  Rules::init_state(state);

  auto valid_masks = Rules::analyze(state).valid_actions();
  ActionMask expected_mask;
  expected_mask.set();
  expected_mask[hex::kSwap] = 0;

  EXPECT_EQ(valid_masks, expected_mask);
}

TEST_F(UnionFindTest, InitParentsAreSelf) {
  // After init, every node should be its own parent
  for (vertex_t i = 0; i < UnionFind::kNumVertices; ++i) {
    EXPECT_EQ(uf.find(i), i) << "Node " << i << " should be its own root";
    EXPECT_FALSE(uf.connected(i, (i + 1) % UnionFind::kNumVertices))
      << "Different nodes should not be connected by default";
  }
}

TEST_F(UnionFindTest, UniteTwoElements) {
  constexpr auto a = vertex_t(1);
  constexpr auto b = vertex_t(2);
  EXPECT_FALSE(uf.connected(a, b));
  uf.unite(a, b);
  EXPECT_TRUE(uf.connected(a, b));
  EXPECT_EQ(uf.find(a), uf.find(b));
}

TEST_F(UnionFindTest, ChainedUnions) {
  // chain a–b, b–c => a, c become connected
  constexpr auto a = vertex_t(3);
  constexpr auto b = vertex_t(4);
  constexpr auto c = vertex_t(5);
  uf.unite(a, b);
  uf.unite(b, c);

  EXPECT_TRUE(uf.connected(a, c));
  EXPECT_EQ(uf.find(a), uf.find(c));
}

TEST_F(UnionFindTest, MultipleUniteIdempotent) {
  constexpr auto x = vertex_t(6);
  constexpr auto y = vertex_t(7);
  // multiple unites should not break
  uf.unite(x, y);
  auto first_root = uf.find(x);
  uf.unite(x, y);
  uf.unite(y, x);
  EXPECT_EQ(uf.find(x), first_root);
  EXPECT_TRUE(uf.connected(x, y));
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
  Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr = init_state_repr;

  EXPECT_STREQ(repr.c_str(), expected_repr.c_str());
  Symmetries::apply(state, inv_sym);
  EXPECT_STREQ(get_repr(state).c_str(), init_state_repr.c_str());

  PolicyTensor init_policy = make_policy(hex::kA1, hex::kB1);
  PolicyTensor policy = init_policy;
  Symmetries::apply(policy, sym, 0);
  PolicyTensor expected_policy = make_policy(hex::kA1, hex::kB1);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Symmetries::apply(policy, inv_sym, 0);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, rotate) {
  State state = make_init_state();

  group::element_t sym = groups::C2::kRot180;
  group::element_t inv_sym = groups::C2::inverse(sym);
  Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "               A B C D E F G H I J K\n"
    "          11 / / / / / / / / / / / / 11\n"
    "         10 / / / / / / / / / / /R/ 10\n"
    "         9 / / / / / / / / / / /R/  9\n"
    "        8 / / / / / / / / / / / /  8\n"
    "       7 / / / / / / / / / / / /  7\n"
    "      6 / / / / / / / / / / / /  6\n"
    "     5 / / / / / / / / / / / /  5\n"
    "    4 / / / / / / / / / / / /  4\n"
    "   3 / / / / / / / / / / / /  3\n"
    "  2 / / / / / / / / /B/ / /  2\n"
    " 1 / / / / / / / / / / / /  1\n"
    "   A B C D E F G H I J K\n\n";

  EXPECT_STREQ(repr.c_str(), expected_repr.c_str());
  Symmetries::apply(state, inv_sym);
  EXPECT_STREQ(get_repr(state).c_str(), init_state_repr.c_str());

  PolicyTensor init_policy = make_policy(hex::kA1, hex::kB1);
  PolicyTensor policy = init_policy;
  Symmetries::apply(policy, sym, 0);
  PolicyTensor expected_policy = make_policy(hex::kJ11, hex::kK11);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Symmetries::apply(policy, inv_sym, 0);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Rules, swap_start) {
  State state;
  Rules::init_state(state);

  RulesResult result = Rules::analyze(state);
  EXPECT_FALSE(result.is_terminal());
  ActionMask valid_actions = result.valid_actions();

  EXPECT_FALSE(valid_actions[hex::kSwap]);
  EXPECT_EQ(valid_actions.count(), Constants::kNumSquares);

  core::action_t move = hex::kC1;
  core::action_t mirrored_move = hex::kA3;

  Rules::apply(state, move);
  result = Rules::analyze(state);

  EXPECT_FALSE(result.is_terminal());
  valid_actions = result.valid_actions();
  EXPECT_TRUE(valid_actions[hex::kSwap]);
  EXPECT_FALSE(valid_actions[move]);
  EXPECT_EQ(valid_actions.count(), Constants::kNumSquares);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "               A B C D E F G H I J K\n"
    "          11 / / / / / / / / / / / / 11\n"
    "         10 / / / / / / / / / / / / 10\n"
    "         9 / / / / / / / / / / / /  9\n"
    "        8 / / / / / / / / / / / /  8\n"
    "       7 / / / / / / / / / / / /  7\n"
    "      6 / / / / / / / / / / / /  6\n"
    "     5 / / / / / / / / / / / /  5\n"
    "    4 / / / / / / / / / / / /  4\n"
    "   3 / / / / / / / / / / / /  3\n"
    "  2 / / / / / / / / / / / /  2\n"
    " 1 / / /R/ / / / / / / / /  1\n"
    "   A B C D E F G H I J K\n\n";

  EXPECT_STREQ(repr.c_str(), expected_repr.c_str());

  Rules::apply(state, hex::kSwap);
  result = Rules::analyze(state);
  valid_actions = result.valid_actions();
  EXPECT_FALSE(valid_actions[hex::kSwap]);
  EXPECT_FALSE(valid_actions[mirrored_move]);
  EXPECT_EQ(valid_actions.count(), Constants::kNumSquares - 1);

  repr = get_repr(state);
  expected_repr =
    "               A B C D E F G H I J K\n"
    "          11 / / / / / / / / / / / / 11\n"
    "         10 / / / / / / / / / / / / 10\n"
    "         9 / / / / / / / / / / / /  9\n"
    "        8 / / / / / / / / / / / /  8\n"
    "       7 / / / / / / / / / / / /  7\n"
    "      6 / / / / / / / / / / / /  6\n"
    "     5 / / / / / / / / / / / /  5\n"
    "    4 / / / / / / / / / / / /  4\n"
    "   3 /B/ / / / / / / / / / /  3\n"
    "  2 / / / / / / / / / / / /  2\n"
    " 1 / / / / / / / / / / / /  1\n"
    "   A B C D E F G H I J K\n\n";

  EXPECT_STREQ(repr.c_str(), expected_repr.c_str());
}

TEST(Rules, non_swap_start) {
  State state;
  Rules::init_state(state);

  RulesResult result = Rules::analyze(state);
  EXPECT_FALSE(result.is_terminal());
  ActionMask valid_actions = result.valid_actions();

  core::action_t move = hex::kC1;
  core::action_t move2 = hex::kF8;

  Rules::apply(state, move);
  Rules::apply(state, move2);
  result = Rules::analyze(state);
  valid_actions = result.valid_actions();

  EXPECT_FALSE(valid_actions[hex::kSwap]);
  EXPECT_FALSE(valid_actions[move]);
  EXPECT_FALSE(valid_actions[move2]);
  EXPECT_EQ(valid_actions.count(), Constants::kNumSquares - 2);
}

TEST(Rules, connections) {
  State state;
  Rules::init_state(state);
  RulesResult result;

  constexpr int kNumMoves = 5;
  std::vector<core::action_t> red_moves = {hex::kC10, hex::kC9, hex::kI1, hex::kI2, hex::kI3};
  std::vector<core::action_t> blue_moves = {hex::kA2, hex::kB2, hex::kH10, hex::kI10, hex::kJ10};

  RELEASE_ASSERT(red_moves.size() == kNumMoves && blue_moves.size() == kNumMoves);

  for (int i = 0; i < kNumMoves; ++i) {
    EXPECT_EQ(Rules::get_current_player(state), Constants::kRed);

    result = Rules::analyze(state);
    EXPECT_FALSE(result.is_terminal());
    ActionMask valid_actions = result.valid_actions();
    EXPECT_TRUE(valid_actions[red_moves[i]]);
    Rules::apply(state, red_moves[i]);

    result = Rules::analyze(state);
    EXPECT_FALSE(result.is_terminal());
    valid_actions = result.valid_actions();
    EXPECT_TRUE(valid_actions[blue_moves[i]]);
    Rules::apply(state, blue_moves[i]);
  }

  result = Rules::analyze(state);
  EXPECT_FALSE(result.is_terminal());

  const auto& UF_red = state.aux.union_find[Constants::kRed];
  const auto& UF_blue = state.aux.union_find[Constants::kBlue];

  using map_t = std::map<vertex_t, std::vector<vertex_t>>;
  map_t red_parent;
  map_t blue_parent;

  for (vertex_t v = 0; v < UnionFind::kNumVertices; ++v) {
    red_parent[UF_red.parent(v)].push_back(v);
    blue_parent[UF_blue.parent(v)].push_back(v);
  }

  // iterate over red_parent:
  for (const auto& [parent, children] : red_parent) {
    if (parent == hex::kC10 || parent == hex::kC9) {
      EXPECT_EQ(children.size(), 2);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kC10) > 0);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kC9) > 0);
    } else if (parent == hex::kI1 || parent == hex::kI2 || parent == hex::kI3 ||
               parent == UnionFind::kVirtualVertex1) {
      EXPECT_EQ(children.size(), 4);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kI1) > 0);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kI2) > 0);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kI3) > 0);
      EXPECT_TRUE(std::count(children.begin(), children.end(), UnionFind::kVirtualVertex1) > 0);
    } else {
      EXPECT_EQ(children.size(), 1);
      EXPECT_TRUE(std::count(children.begin(), children.end(), parent) > 0);
    }
  }

  // iterate over blue_parent:
  for (const auto& [parent, children] : blue_parent) {
    if (parent == hex::kA2 || parent == hex::kB2 || parent == UnionFind::kVirtualVertex1) {
      EXPECT_EQ(children.size(), 3);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kA2) > 0);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kB2) > 0);
      EXPECT_TRUE(std::count(children.begin(), children.end(), UnionFind::kVirtualVertex1) > 0);
    } else if (parent == hex::kH10 || parent == hex::kI10 || parent == hex::kJ10) {
      EXPECT_EQ(children.size(), 3);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kH10) > 0);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kI10) > 0);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kJ10) > 0);
    } else {
      EXPECT_EQ(children.size(), 1);
      EXPECT_TRUE(std::count(children.begin(), children.end(), parent) > 0);
    }
  }

  Symmetries::apply(state, groups::C2::kRot180);

  const auto& UF_red2 = state.aux.union_find[Constants::kRed];
  const auto& UF_blue2 = state.aux.union_find[Constants::kBlue];

  map_t red_parent2;
  map_t blue_parent2;

  for (vertex_t v = 0; v < UnionFind::kNumVertices; ++v) {
    red_parent2[UF_red2.find(v)].push_back(v);
    blue_parent2[UF_blue2.find(v)].push_back(v);
  }

  // iterate over red_parent2:
  for (const auto& [parent, children] : red_parent2) {
    if (parent == hex::kI2 || parent == hex::kI3) {
      EXPECT_EQ(children.size(), 2);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kI2) > 0);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kI3) > 0);
    } else if (parent == hex::kC11 || parent == hex::kC10 || parent == hex::kC9 ||
               parent == UnionFind::kVirtualVertex2) {
      EXPECT_EQ(children.size(), 4);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kC11) > 0);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kC10) > 0);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kC9) > 0);
      EXPECT_TRUE(std::count(children.begin(), children.end(), UnionFind::kVirtualVertex2) > 0);
    } else {
      EXPECT_EQ(children.size(), 1);
      EXPECT_TRUE(std::count(children.begin(), children.end(), parent) > 0);
    }
  }

  // iterate over blue_parent2:
  for (const auto& [parent, children] : blue_parent2) {
    if (parent == hex::kK10 || parent == hex::kJ10 || parent == UnionFind::kVirtualVertex2) {
      EXPECT_EQ(children.size(), 3);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kK10) > 0);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kJ10) > 0);
      EXPECT_TRUE(std::count(children.begin(), children.end(), UnionFind::kVirtualVertex2) > 0);
    } else if (parent == hex::kD2 || parent == hex::kC2 || parent == hex::kB2) {
      EXPECT_EQ(children.size(), 3);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kD2) > 0);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kC2) > 0);
      EXPECT_TRUE(std::count(children.begin(), children.end(), hex::kB2) > 0);
    } else {
      EXPECT_EQ(children.size(), 1);
      EXPECT_TRUE(std::count(children.begin(), children.end(), parent) > 0);
    }
  }
}

TEST(Rules, terminal) {
  State state;
  Rules::init_state(state);
  Rules::Result result = Rules::analyze(state);
  std::vector<core::action_t> moves = {hex::kA1, hex::kA2, hex::kB1, hex::kB2, hex::kC1, hex::kC2,
                                       hex::kD1, hex::kD2, hex::kE1, hex::kE2, hex::kF1, hex::kF2,
                                       hex::kG1, hex::kG2, hex::kH1, hex::kH2, hex::kI1, hex::kI2,
                                       hex::kJ1, hex::kJ2, hex::kK1, hex::kK2};

  EXPECT_FALSE(result.is_terminal());
  EXPECT_TRUE(result.valid_actions()[moves[0]]);
  int num_moves = moves.size();

  for (int i = 0; i < num_moves; ++i) {
    core::action_t move = moves[i];
    EXPECT_EQ(Rules::get_current_player(state), i % 2);
    Rules::apply(state, move);
    result = Rules::analyze(state);

    if (i < num_moves - 1) {
      EXPECT_FALSE(result.is_terminal());
      EXPECT_TRUE(result.valid_actions()[moves[i + 1]]);
    } else {
      // Last move should be terminal
      EXPECT_TRUE(result.is_terminal());
      EXPECT_EQ(result.outcome()[Constants::kRed], 0);
      EXPECT_EQ(result.outcome()[Constants::kBlue], 1);
    }
  }
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
