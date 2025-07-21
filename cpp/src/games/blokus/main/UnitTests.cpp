#include "games/blokus/Game.hpp"
#include "games/blokus/Types.hpp"
#include "util/GTestUtil.hpp"

#include <gtest/gtest.h>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using namespace blokus;

using State = Game::State;
using StateHistory = Game::StateHistory;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;

template <typename B>
std::string get_repr(const B& board) {
  BoardString s;
  s.set(board, dTimes);

  std::ostringstream ss;
  s.print(ss, true);
  return ss.str();
}

std::string get_repr(const State& state) {
  std::ostringstream ss;
  Game::IO::print_state(ss, state, kPass+1);
  std::string s = ss.str();

  std::vector<std::string> lines;
  std::istringstream iss(s);
  std::string line;
  while (std::getline(iss, line)) {
    lines.push_back(line);
  }

  std::string repr;
  for (int i = 0; i < 22; ++i) {
    repr += lines[i];
    repr += '\n';
  }

  return repr;
}

std::string make_full_piece_repr(PieceOrientationCorner poc, Location loc) {
  BitBoardSlice mask = poc.to_bitboard_mask(loc);
  BitBoardSlice adj_mask = poc.to_adjacent_bitboard_mask(loc);
  BitBoardSlice diag_mask = poc.to_diagonal_bitboard_mask(loc);

  BoardString s;
  s.set(mask, dCircle);
  s.set(adj_mask, dPlus);
  s.set(diag_mask, dStar);

  std::ostringstream ss;
  s.print(ss, true);
  return ss.str();
}

// Return true if the test passes
void full_piece_repr_test(PieceOrientationCorner poc, int8_t row, int8_t col,
                          const std::string& expected_repr) {
  std::string actual_repr = make_full_piece_repr(poc, Location{row, col});
  EXPECT_EQ(actual_repr, expected_repr);
}

TEST(Location, flatten) {
  Location invalid_loc{-1, -1};
  Location loc1{0, 0};
  Location loc2{2, 13};

  EXPECT_FALSE(invalid_loc.valid());

  for (auto loc : {loc1, loc2}) {
    EXPECT_TRUE(loc.valid());
  }

  int f1 = loc1.flatten();
  int f2 = loc2.flatten();

  EXPECT_EQ(f1, 0);
  EXPECT_EQ(f2, 53);
  EXPECT_EQ(Location::unflatten(f1), loc1);
  EXPECT_EQ(Location::unflatten(f2), loc2);
}

TEST(BitBoard, all) {
  BitBoard board;
  board.clear();

  board.set(0, 0);
  board.set(Location{0, 19});

  std::string actual_repr = get_repr(board);
  std::string expected_repr =
      "   ABCDEFGHIJKLMNOPQRST\n"
      " 1 x..................x  1\n"
      "   ABCDEFGHIJKLMNOPQRST\n";

  EXPECT_EQ(actual_repr, expected_repr);

  // |

  BitBoard board2;
  board2.clear();

  board2.set(0, 0);
  board2.set(2, 4);
  board2.set(Location{5, 1});

  BitBoard board3 = board | board2;

  std::string actual_repr3 = get_repr(board3);
  std::string expected_repr3 =
      "   ABCDEFGHIJKLMNOPQRST\n"
      " 6 .x..................  6\n"
      " 3 ....x...............  3\n"
      " 1 x..................x  1\n"
      "   ABCDEFGHIJKLMNOPQRST\n";

  EXPECT_EQ(actual_repr3, expected_repr3);

  // &=

  BitBoard board4 = board;
  board4 &= board2;

  std::string actual_repr4 = get_repr(board4);
  std::string expected_repr4 =
      "   ABCDEFGHIJKLMNOPQRST\n"
      " 1 x...................  1\n"
      "   ABCDEFGHIJKLMNOPQRST\n";

  EXPECT_EQ(actual_repr4, expected_repr4);

  // ~

  BitBoard board5 = ~board;
  std::string actual_repr5 = get_repr(board5);
  std::string expected_repr5 =
      "   ABCDEFGHIJKLMNOPQRST\n"
      "20 xxxxxxxxxxxxxxxxxxxx 20\n"
      "19 xxxxxxxxxxxxxxxxxxxx 19\n"
      "18 xxxxxxxxxxxxxxxxxxxx 18\n"
      "17 xxxxxxxxxxxxxxxxxxxx 17\n"
      "16 xxxxxxxxxxxxxxxxxxxx 16\n"
      "15 xxxxxxxxxxxxxxxxxxxx 15\n"
      "14 xxxxxxxxxxxxxxxxxxxx 14\n"
      "13 xxxxxxxxxxxxxxxxxxxx 13\n"
      "12 xxxxxxxxxxxxxxxxxxxx 12\n"
      "11 xxxxxxxxxxxxxxxxxxxx 11\n"
      "10 xxxxxxxxxxxxxxxxxxxx 10\n"
      " 9 xxxxxxxxxxxxxxxxxxxx  9\n"
      " 8 xxxxxxxxxxxxxxxxxxxx  8\n"
      " 7 xxxxxxxxxxxxxxxxxxxx  7\n"
      " 6 xxxxxxxxxxxxxxxxxxxx  6\n"
      " 5 xxxxxxxxxxxxxxxxxxxx  5\n"
      " 4 xxxxxxxxxxxxxxxxxxxx  4\n"
      " 3 xxxxxxxxxxxxxxxxxxxx  3\n"
      " 2 xxxxxxxxxxxxxxxxxxxx  2\n"
      " 1 .xxxxxxxxxxxxxxxxxx.  1\n"
      "   ABCDEFGHIJKLMNOPQRST\n";

  EXPECT_EQ(actual_repr5, expected_repr5);

  // any()

  BitBoard empty_board;
  empty_board.clear();

  EXPECT_FALSE(empty_board.any());
  EXPECT_TRUE(board.any());
  EXPECT_EQ(board.count(), 2);
  EXPECT_EQ(board5.count(), 398);

  // clear_at_and_after()

  BitBoard board6 = board5;
  board6.clear_at_and_after(Location{5, 3});

  std::string actual_repr6 = get_repr(board6);
  std::string expected_repr6 =
      "   ABCDEFGHIJKLMNOPQRST\n"
      " 6 xxx.................  6\n"
      " 5 xxxxxxxxxxxxxxxxxxxx  5\n"
      " 4 xxxxxxxxxxxxxxxxxxxx  4\n"
      " 3 xxxxxxxxxxxxxxxxxxxx  3\n"
      " 2 xxxxxxxxxxxxxxxxxxxx  2\n"
      " 1 .xxxxxxxxxxxxxxxxxx.  1\n"
      "   ABCDEFGHIJKLMNOPQRST\n";

  EXPECT_EQ(actual_repr6, expected_repr6);

  // get_row()

  int get_row_k[] = {0, 4, 5, 6};
  uint32_t get_row_expected[] = {0b01111111111111111110, 0b11111111111111111111, 0b111, 0};
  for (int i = 0; i < 4; i++) {
    uint32_t actual = board6.get_row(get_row_k[i]);
    uint32_t expected = get_row_expected[i];
    EXPECT_EQ(actual, expected);
  }

  // get()
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      bool actual = board6.get(row, col);
      bool expected = row > 0 && row < 5;
      if (row == 0) {
        expected = col > 0 && col < 19;
      } else if (row == 5) {
        expected = col < 3;
      }
      EXPECT_EQ(actual, expected);
    }
  }

  // get_set_locations()
  constexpr int kNumExpectedLocations = 4;
  Location expected_locations[kNumExpectedLocations] = {{0, 0}, {0, 19}, {2, 4}, {5, 1}};

  int i = 0;
  for (Location actual_loc : board3.get_set_locations()) {
    EXPECT_LT(i, kNumExpectedLocations);
    Location expected_loc = expected_locations[i];
    EXPECT_EQ(actual_loc, expected_loc);
    ++i;
  }
  EXPECT_EQ(i, kNumExpectedLocations);

  // write_to()

  std::bitset<kNumCells> expected_bitset;
  for (i = 0; i < kNumExpectedLocations; ++i) {
    expected_bitset[expected_locations[i].flatten()] = true;
  }
  std::bitset<kNumCells> actual_bitset;
  board3.write_to(actual_bitset);
  EXPECT_EQ(actual_bitset, expected_bitset);

  // get_corner_constraint()

  BitBoard board7;
  board7.clear();
  board7.set(0, 1);
  board7.set(0, 3);
  board7.set(1, 0);
  board7.set(1, 2);
  board7.set(1, 4);
  board7.set(2, 3);

  Location constraint_locs7[8] = {{0, 0}, {0, 2}, {0, 4}, {1, 1}, {1, 3}, {2, 0}, {2, 2}, {2, 4}};
  corner_constraint_t expected_constraints7[8] = {ccNone, ccNone, ccE,  ccN,
                                                  ccNone, ccNE,   ccNW, ccNE};

  for (i = 0; i < 8; ++i) {
    corner_constraint_t actual = board7.get_corner_constraint(constraint_locs7[i]);
    corner_constraint_t expected = expected_constraints7[i];
    EXPECT_EQ(actual, expected);
  }

  BitBoard board8;
  board8.clear();
  board8.set(0, 18);
  board8.set(0, 16);
  board8.set(1, 19);
  board8.set(1, 17);
  board8.set(1, 15);
  board8.set(2, 16);

  Location constraint_locs8[8] = {{0, 19}, {0, 17}, {0, 15}, {1, 18},
                                  {1, 16}, {2, 19}, {2, 17}, {2, 15}};
  corner_constraint_t expected_constraints8[8] = {ccNone, ccNone, ccW,  ccN,
                                                  ccNone, ccNW,   ccNE, ccNW};

  for (i = 0; i < 8; ++i) {
    corner_constraint_t actual = board8.get_corner_constraint(constraint_locs8[i]);
    corner_constraint_t expected = expected_constraints8[i];
    EXPECT_EQ(actual, expected);
  }

  BitBoard board9;
  board9.clear();
  board9.set(19, 1);
  board9.set(19, 3);
  board9.set(18, 0);
  board9.set(18, 2);
  board9.set(18, 4);
  board9.set(17, 3);

  Location constraint_locs9[8] = {{19, 0}, {19, 2}, {19, 4}, {18, 1},
                                  {18, 3}, {17, 0}, {17, 2}, {17, 4}};
  corner_constraint_t expected_constraints9[8] = {ccNone, ccNone, ccE,  ccS,
                                                  ccNone, ccSE,   ccSW, ccSE};

  for (i = 0; i < 8; ++i) {
    corner_constraint_t actual = board9.get_corner_constraint(constraint_locs9[i]);
    corner_constraint_t expected = expected_constraints9[i];
    EXPECT_EQ(actual, expected);
  }

  BitBoard board10;
  board10.clear();
  board10.set(19, 18);
  board10.set(19, 16);
  board10.set(18, 19);
  board10.set(18, 17);
  board10.set(18, 15);
  board10.set(17, 16);

  Location constraint_locs10[8] = {{19, 19}, {19, 17}, {19, 15}, {18, 18},
                                  {18, 16}, {17, 19}, {17, 17}, {17, 15}};
  corner_constraint_t expected_constraints10[8] = {ccNone, ccNone, ccW,  ccS,
                                                  ccNone, ccSW,   ccSE, ccSW};

  for (i = 0; i < 8; ++i) {
    corner_constraint_t actual = board10.get_corner_constraint(constraint_locs10[i]);
    corner_constraint_t expected = expected_constraints10[i];
    EXPECT_EQ(actual, expected);
  }

  // |=

  // 27: O4/0
  //
  // *..*
  // .oo.
  // .ox.
  // *..*
  PieceOrientationCorner poc(27);
  BitBoardSlice slice = poc.to_bitboard_mask(Location{0, 1});  // snug against bottom left corner
  BitBoard board11 = board;
  board11 |= slice;

  std::string actual_repr11 = get_repr(board11);
  std::string expected_repr11 =
      "   ABCDEFGHIJKLMNOPQRST\n"
      " 2 xx..................  2\n"
      " 1 xx.................x  1\n"
      "   ABCDEFGHIJKLMNOPQRST\n";

  EXPECT_EQ(actual_repr11, expected_repr11);

  // intersects()
  EXPECT_TRUE(board.intersects(slice));

  BitBoard board12;
  board12.clear();
  EXPECT_FALSE(board12.intersects(slice));
}

TEST(PieceOrientationCorner, all) {
  // 272: Y5/1
  //
  // *....*
  // .xooo.
  // *.o..*
  //  *.*
  PieceOrientationCorner poc(272);

  Piece p = poc.to_piece();
  PieceOrientation po = poc.to_piece_orientation();
  EXPECT_EQ(p, 19);
  EXPECT_EQ(po, 80);

  Location expected_loc{2, 1};
  Location actual_loc = poc.corner_offset();

  EXPECT_EQ(actual_loc, expected_loc);

  BitBoardSlice mask_0_0 = poc.to_bitboard_mask(Location{0, 0});
  BitBoardSlice mask_0_19 = poc.to_bitboard_mask(Location{0, 19});
  BitBoardSlice mask_19_0 = poc.to_bitboard_mask(Location{19, 0});
  BitBoardSlice mask_19_19 = poc.to_bitboard_mask(Location{19, 19});

  EXPECT_TRUE(mask_0_0.empty());
  EXPECT_TRUE(mask_0_19.empty());
  EXPECT_FALSE(mask_19_0.empty());
  EXPECT_TRUE(mask_19_19.empty());
  full_piece_repr_test(poc, 19, 0,
                       "   ABCDEFGHIJKLMNOPQRST\n"
                       "20 oooo+............... 20\n"
                       "19 +o++*............... 19\n"
                       "18 *+*................. 18\n"
                       "   ABCDEFGHIJKLMNOPQRST\n");

  full_piece_repr_test(poc, 19, 1,
                       "   ABCDEFGHIJKLMNOPQRST\n"
                       "20 +oooo+.............. 20\n"
                       "19 *+o++*.............. 19\n"
                       "18 .*+*................ 18\n"
                       "   ABCDEFGHIJKLMNOPQRST\n");

  full_piece_repr_test(poc, 18, 0,
                       "   ABCDEFGHIJKLMNOPQRST\n"
                       "20 ++++*............... 20\n"
                       "19 oooo+............... 19\n"
                       "18 +o++*............... 18\n"
                       "17 *+*................. 17\n"
                       "   ABCDEFGHIJKLMNOPQRST\n");

  full_piece_repr_test(poc, 18, 1,
                       "   ABCDEFGHIJKLMNOPQRST\n"
                       "20 *++++*.............. 20\n"
                       "19 +oooo+.............. 19\n"
                       "18 *+o++*.............. 18\n"
                       "17 .*+*................ 17\n"
                       "   ABCDEFGHIJKLMNOPQRST\n");

  full_piece_repr_test(poc, 1, 0,
                       "   ABCDEFGHIJKLMNOPQRST\n"
                       " 3 ++++*...............  3\n"
                       " 2 oooo+...............  2\n"
                       " 1 +o++*...............  1\n"
                       "   ABCDEFGHIJKLMNOPQRST\n");

  full_piece_repr_test(poc, 1, 1,
                       "   ABCDEFGHIJKLMNOPQRST\n"
                       " 3 *++++*..............  3\n"
                       " 2 +oooo+..............  2\n"
                       " 1 *+o++*..............  1\n"
                       "   ABCDEFGHIJKLMNOPQRST\n");

  full_piece_repr_test(poc, 2, 0,
                       "   ABCDEFGHIJKLMNOPQRST\n"
                       " 4 ++++*...............  4\n"
                       " 3 oooo+...............  3\n"
                       " 2 +o++*...............  2\n"
                       " 1 *+*.................  1\n"
                       "   ABCDEFGHIJKLMNOPQRST\n");

  full_piece_repr_test(poc, 2, 1,
                       "   ABCDEFGHIJKLMNOPQRST\n"
                       " 4 *++++*..............  4\n"
                       " 3 +oooo+..............  3\n"
                       " 2 *+o++*..............  2\n"
                       " 1 .*+*................  1\n"
                       "   ABCDEFGHIJKLMNOPQRST\n");

  full_piece_repr_test(poc, 1, 16,
                       "   ABCDEFGHIJKLMNOPQRST\n"
                       " 3 ...............*++++  3\n"
                       " 2 ...............+oooo  2\n"
                       " 1 ...............*+o++  1\n"
                       "   ABCDEFGHIJKLMNOPQRST\n");

  full_piece_repr_test(poc, 1, 15,
                       "   ABCDEFGHIJKLMNOPQRST\n"
                       " 3 ..............*++++*  3\n"
                       " 2 ..............+oooo+  2\n"
                       " 1 ..............*+o++*  1\n"
                       "   ABCDEFGHIJKLMNOPQRST\n");

  full_piece_repr_test(poc, 2, 16,
                       "   ABCDEFGHIJKLMNOPQRST\n"
                       " 4 ...............*++++  4\n"
                       " 3 ...............+oooo  3\n"
                       " 2 ...............*+o++  2\n"
                       " 1 ................*+*.  1\n"
                       "   ABCDEFGHIJKLMNOPQRST\n");

  full_piece_repr_test(poc, 2, 15,
                       "   ABCDEFGHIJKLMNOPQRST\n"
                       " 4 ..............*++++*  4\n"
                       " 3 ..............+oooo+  3\n"
                       " 2 ..............*+o++*  2\n"
                       " 1 ...............*+*..  1\n"
                       "   ABCDEFGHIJKLMNOPQRST\n");

  full_piece_repr_test(poc, 19, 16,
                       "   ABCDEFGHIJKLMNOPQRST\n"
                       "20 ...............+oooo 20\n"
                       "19 ...............*+o++ 19\n"
                       "18 ................*+*. 18\n"
                       "   ABCDEFGHIJKLMNOPQRST\n");

  full_piece_repr_test(poc, 19, 15,
                       "   ABCDEFGHIJKLMNOPQRST\n"
                       "20 ..............+oooo+ 20\n"
                       "19 ..............*+o++* 19\n"
                       "18 ...............*+*.. 18\n"
                       "   ABCDEFGHIJKLMNOPQRST\n");

  full_piece_repr_test(poc, 18, 16,
                       "   ABCDEFGHIJKLMNOPQRST\n"
                       "20 ...............*++++ 20\n"
                       "19 ...............+oooo 19\n"
                       "18 ...............*+o++ 18\n"
                       "17 ................*+*. 17\n"
                       "   ABCDEFGHIJKLMNOPQRST\n");

  full_piece_repr_test(poc, 18, 15,
                       "   ABCDEFGHIJKLMNOPQRST\n"
                       "20 ..............*++++* 20\n"
                       "19 ..............+oooo+ 19\n"
                       "18 ..............*+o++* 18\n"
                       "17 ...............*+*.. 17\n"
                       "   ABCDEFGHIJKLMNOPQRST\n");
}

TEST(State, load) {
  std::string repr =
      "   ABCDEFGHIJKLMNOPQRST\n"
      "20 Y...Y..Y..YY.R....RR 20\n"
      "19 Y...Y...YY.YYRR...RR 19\n"
      "18 .Y..Y..YY.R...RR..R. 18\n"
      "17 .YYY...Y.YRRR...RR.R 17\n"
      "16 BBY.YYY.BY...RRR..RR 16\n"
      "15 .B.Y.YB.BYR....R.... 15\n"
      "14 BB.YYBBB.YRR.R.R.RRR 14\n"
      "13 ..BBB.YY.YR.RRR.R... 13\n"
      "12 BB.YBBYYY.R....RRR.. 12\n"
      "11 B.BYYY...Y..R.R.R.R. 11\n"
      "10 B.BB..YYYY..RRR....R 10\n"
      " 9 B.YYYYBRRRRR..G.GG.R  9\n"
      " 8 .BB.BBBB.BBBB.G..G.R  8\n"
      " 7 B.BB.GG.B..G.GG.G..R  7\n"
      " 6 GGGBG.G.BGGG.G.GGGG.  6\n"
      " 5 BBBGGBGGB..G..G....G  5\n"
      " 4 B..G.BBB..G...G...GG  4\n"
      " 3 .BG.G.B.GGG...G..GG.  3\n"
      " 2 BBG.GGGG.G.GG.G.....  2\n"
      " 1 BB.G.......GGG.GGGGG  1\n"
      "   ABCDEFGHIJKLMNOPQRST\n";

  State state = Game::IO::load(repr);
  std::string repr2 = get_repr(state);

  EXPECT_EQ(repr, repr2);
}

int main(int argc, char** argv) {
  return launch_gtest(argc, argv);
}
