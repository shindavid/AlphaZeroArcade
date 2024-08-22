#include <games/blokus/Game.hpp>
#include <games/blokus/Types.hpp>

int global_pass_count = 0;
int global_fail_count = 0;

using namespace blokus;

using BaseState = Game::BaseState;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;

template<typename B>
std::string get_repr(const B& board) {
  BoardString s;
  s.set(board, "x");

  std::ostringstream ss;
  s.print(ss);
  return ss.str();
}

std::string make_full_piece_repr(PieceOrientationCorner poc, Location loc) {
  BitBoardSlice mask = poc.to_bitboard_mask(loc);
  BitBoardSlice adj_mask = poc.to_adjacent_bitboard_mask(loc);
  BitBoardSlice diag_mask = poc.to_diagonal_bitboard_mask(loc);

  BoardString s;
  s.set(mask, "o");
  s.set(adj_mask, "+");
  s.set(diag_mask, "*");

  std::ostringstream ss;
  s.print(ss, true);
  return ss.str();
}

/*
 * Validates that the non-trivial lines of actual_repr match expected_repr.
 *
 * Returns true if the test passes. Else, prints the failure, increments global_fail_count, and
 * returns false.
 */
bool validate_repr(const char* func, int line, const std::string& actual_repr,
                   const std::string& expected_repr) {
  if (actual_repr != expected_repr) {
    printf("Failure at %s():%d\n", func, line);
    std::cout << "Expected:" << std::endl;
    std::cout << expected_repr << std::endl;
    std::cout << "But got:" << std::endl;
    std::cout << actual_repr << std::endl;
    global_fail_count++;
    return false;
  } else {
    return true;
  }
}

// Return true if the test passes
bool full_piece_repr_test(const char* file, int line, PieceOrientationCorner poc, int8_t row,
                          int8_t col, const std::string& expected_repr) {
  std::string actual_repr = make_full_piece_repr(poc, Location{row, col});
  return validate_repr(file, line, actual_repr, expected_repr);
}

void test_location() {
  Location invalid_loc{-1, -1};
  Location loc1{0, 0};
  Location loc2{2, 13};

  if (invalid_loc.valid()) {
    global_fail_count++;
    std::cout << "Invalid location is valid" << std::endl;
    return;
  }

  for (auto loc : {loc1, loc2}) {
    if (!loc.valid()) {
      global_fail_count++;
      std::cout << "Valid location is invalid" << std::endl;
      return;
    }
  }

  int f1 = loc1.flatten();
  int f2 = loc2.flatten();

  if (f1 != 0 || f2 != 53) {
    global_fail_count++;
    std::cout << "Invalid Location::flatten()" << std::endl;
    return;
  }

  if (Location::unflatten(f1) != loc1) {
    global_fail_count++;
    std::cout << "Invalid Location::unflatten()" << std::endl;
    return;
  }

  if (Location::unflatten(f2) != loc2) {
    global_fail_count++;
    std::cout << "Invalid Location::unflatten()" << std::endl;
    return;
  }

  printf("Success: %s()\n", __func__);
  global_pass_count++;
}

void test_bit_board() {
  BitBoard board;
  board.clear();

  board.set(0, 0);
  board.set(Location{0, 19});

  std::string actual_repr = get_repr(board);
  std::string expected_repr =
      "   ABCDEFGHIJKLMNOPQRST\n"
      " 1 x..................x  1\n"
      "   ABCDEFGHIJKLMNOPQRST\n";

  if (!validate_repr(__func__, __LINE__, actual_repr, expected_repr)) {
    return;
  }

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

  if (!validate_repr(__func__, __LINE__, actual_repr3, expected_repr3)) {
    return;
  }

  // &=

  BitBoard board4 = board;
  board4 &= board2;

  std::string actual_repr4 = get_repr(board4);
  std::string expected_repr4 =
      "   ABCDEFGHIJKLMNOPQRST\n"
      " 1 x...................  1\n"
      "   ABCDEFGHIJKLMNOPQRST\n";

  if (!validate_repr(__func__, __LINE__, actual_repr4, expected_repr4)) {
    return;
  }

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

  if (!validate_repr(__func__, __LINE__, actual_repr5, expected_repr5)) {
    return;
  }

  // any()

  BitBoard empty_board;
  empty_board.clear();

  if (empty_board.any()) {
    global_fail_count++;
    std::cout << "Empty board is not empty" << std::endl;
    return;
  }

  if (!board.any()) {
    global_fail_count++;
    std::cout << "Non-empty board is empty" << std::endl;
    return;
  }

  if (board.count() != 2) {
    global_fail_count++;
    std::cout << "BitBoard::count() failure" << std::endl;
    return;
  }

  if (board5.count() != 398) {
    global_fail_count++;
    std::cout << "BitBoard::count() failure" << std::endl;
    return;
  }

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

  if (!validate_repr(__func__, __LINE__, actual_repr6, expected_repr6)) {
    return;
  }

  // get_row()

  int get_row_k[] = {0, 4, 5, 6};
  uint32_t get_row_expected[] = {0b01111111111111111110, 0b11111111111111111111, 0b111, 0};
  for (int i = 0; i < 4; i++) {
    uint32_t actual = board6.get_row(get_row_k[i]);
    uint32_t expected = get_row_expected[i];
    if (actual != expected) {
      global_fail_count++;
      printf("board6.get_row() failure i=%d (%u != %u)\n", i, actual, expected);
      return;
    }
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
      if (actual != expected) {
        global_fail_count++;
        printf("board6.get() failure row=%d col=%d (%d != %d)\n", row, col, actual, expected);
        return;
      }
    }
  }

  // get_set_locations()
  constexpr int kNumExpectedLocations = 4;
  Location expected_locations[kNumExpectedLocations] = {{0, 0}, {0, 19}, {2, 4}, {5, 1}};

  int i = 0;
  for (Location actual_loc : board3.get_set_locations()) {
    if (i >= kNumExpectedLocations) {
      global_fail_count++;
      printf("board3.get_set_locations() failure: too many locations\n");
      return;
    }
    Location expected_loc = expected_locations[i];
    if (actual_loc != expected_loc) {
      global_fail_count++;
      printf("board3.get_set_locations() failure: location i=%d (%d, %d) != (%d, %d)\n", i,
             (int)actual_loc.row, (int)actual_loc.col, (int)expected_loc.row,
             (int)expected_loc.col);
      return;
    }
    ++i;
  }
  if (i < kNumExpectedLocations) {
    global_fail_count++;
    printf("board3.get_set_locations() failure: too few locations\n");
    return;
  }

  // write_to()

  std::bitset<kNumCells> expected_bitset;
  for (i = 0; i < kNumExpectedLocations; ++i) {
    expected_bitset[expected_locations[i].flatten()] = true;
  }
  std::bitset<kNumCells> actual_bitset;
  board3.write_to(actual_bitset);
  if (actual_bitset != expected_bitset) {
    global_fail_count++;
    printf("board3.write_to() failure\n");
    return;
  }

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
    if (actual != expected) {
      global_fail_count++;
      printf("board7.get_corner_constraint() failure i=%d (%d != %d)\n", i, (int)actual,
             (int)expected);
      return;
    }
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
    if (actual != expected) {
      global_fail_count++;
      printf("board8.get_corner_constraint() failure i=%d (%d != %d)\n", i, (int)actual,
             (int)expected);
      return;
    }
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
    if (actual != expected) {
      global_fail_count++;
      printf("board9.get_corner_constraint() failure i=%d (%d != %d)\n", i, (int)actual,
             (int)expected);
      return;
    }
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
    if (actual != expected) {
      global_fail_count++;
      printf("board10.get_corner_constraint() failure i=%d (%d != %d)\n", i, (int)actual,
             (int)expected);
      return;
    }
  }

  printf("Success: %s()\n", __func__);
  global_pass_count++;
}

void test_piece_orientation_corner() {
  // 272: Y5/1
  //
  // *....*
  // .xooo.
  // *.o..*
  //  *.*
  PieceOrientationCorner poc(272);

  Piece p = poc.to_piece();
  PieceOrientation po = poc.to_piece_orientation();

  if (p != 19) {
    global_fail_count++;
    printf("PieceOrientationCorner::to_piece() failure (%d)\n", int(p));
    return;
  }

  if (po != 80) {
    global_fail_count++;
    printf("PieceOrientationCorner::to_piece_orientation() failure (%d)\n", int(po));
    return;
  }

  Location expected_loc{2, 1};
  Location actual_loc = poc.corner_offset();

  if (actual_loc != expected_loc) {
    global_fail_count++;
    printf("PieceOrientationCorner::corner_offset() failure (%d, %d != %d, %d)\n",
           (int)actual_loc.row, (int)actual_loc.col, (int)expected_loc.row, (int)expected_loc.col);
    return;
  }

  BitBoardSlice mask_0_0 = poc.to_bitboard_mask(Location{0, 0});
  BitBoardSlice mask_0_19 = poc.to_bitboard_mask(Location{0, 19});
  BitBoardSlice mask_19_0 = poc.to_bitboard_mask(Location{19, 0});
  BitBoardSlice mask_19_19 = poc.to_bitboard_mask(Location{19, 19});

  if (!mask_0_0.empty()) {
    global_fail_count++;
    printf("PieceOrientationCorner::to_bitboard_mask() failure (0, 0) - expected empty\n");
    return;
  }

  if (!mask_0_19.empty()) {
    global_fail_count++;
    printf("PieceOrientationCorner::to_bitboard_mask() failure (0, 19) - expected empty\n");
    return;
  }

  if (mask_19_0.empty()) {
    global_fail_count++;
    printf("PieceOrientationCorner::to_bitboard_mask() failure (19, 0) - expected nonempty\n");
    return;
  }

  if (!mask_19_19.empty()) {
    global_fail_count++;
    printf("PieceOrientationCorner::to_bitboard_mask() failure (0, 19) - expected empty\n");
    return;
  }

  if (!full_piece_repr_test(__FILE__, __LINE__, poc, 19, 0,
                            "   ABCDEFGHIJKLMNOPQRST\n"
                            "20 oooo+............... 20\n"
                            "19 +o++*............... 19\n"
                            "18 *+*................. 18\n"
                            "   ABCDEFGHIJKLMNOPQRST\n")) {
    return;
  }

  if (!full_piece_repr_test(__FILE__, __LINE__, poc, 19, 1,
                            "   ABCDEFGHIJKLMNOPQRST\n"
                            "20 +oooo+.............. 20\n"
                            "19 *+o++*.............. 19\n"
                            "18 .*+*................ 18\n"
                            "   ABCDEFGHIJKLMNOPQRST\n")) {
    return;
  }

  if (!full_piece_repr_test(__FILE__, __LINE__, poc, 18, 0,
                            "   ABCDEFGHIJKLMNOPQRST\n"
                            "20 ++++*............... 20\n"
                            "19 oooo+............... 19\n"
                            "18 +o++*............... 18\n"
                            "17 *+*................. 17\n"
                            "   ABCDEFGHIJKLMNOPQRST\n")) {
    return;
  }

  if (!full_piece_repr_test(__FILE__, __LINE__, poc, 18, 1,
                            "   ABCDEFGHIJKLMNOPQRST\n"
                            "20 *++++*.............. 20\n"
                            "19 +oooo+.............. 19\n"
                            "18 *+o++*.............. 18\n"
                            "17 .*+*................ 17\n"
                            "   ABCDEFGHIJKLMNOPQRST\n")) {
    return;
  }

  if (!full_piece_repr_test(__FILE__, __LINE__, poc, 1, 0,
                            "   ABCDEFGHIJKLMNOPQRST\n"
                            " 3 ++++*...............  3\n"
                            " 2 oooo+...............  2\n"
                            " 1 +o++*...............  1\n"
                            "   ABCDEFGHIJKLMNOPQRST\n")) {
    return;
  }

  if (!full_piece_repr_test(__FILE__, __LINE__, poc, 1, 1,
                            "   ABCDEFGHIJKLMNOPQRST\n"
                            " 3 *++++*..............  3\n"
                            " 2 +oooo+..............  2\n"
                            " 1 *+o++*..............  1\n"
                            "   ABCDEFGHIJKLMNOPQRST\n")) {
    return;
  }

  if (!full_piece_repr_test(__FILE__, __LINE__, poc, 2, 0,
                            "   ABCDEFGHIJKLMNOPQRST\n"
                            " 4 ++++*...............  4\n"
                            " 3 oooo+...............  3\n"
                            " 2 +o++*...............  2\n"
                            " 1 *+*.................  1\n"
                            "   ABCDEFGHIJKLMNOPQRST\n")) {
    return;
  }

  if (!full_piece_repr_test(__FILE__, __LINE__, poc, 2, 1,
                            "   ABCDEFGHIJKLMNOPQRST\n"
                            " 4 *++++*..............  4\n"
                            " 3 +oooo+..............  3\n"
                            " 2 *+o++*..............  2\n"
                            " 1 .*+*................  1\n"
                            "   ABCDEFGHIJKLMNOPQRST\n")) {
    return;
  }

  if (!full_piece_repr_test(__FILE__, __LINE__, poc, 1, 16,
                            "   ABCDEFGHIJKLMNOPQRST\n"
                            " 3 ...............*++++  3\n"
                            " 2 ...............+oooo  2\n"
                            " 1 ...............*+o++  1\n"
                            "   ABCDEFGHIJKLMNOPQRST\n")) {
    return;
  }

  if (!full_piece_repr_test(__FILE__, __LINE__, poc, 1, 15,
                            "   ABCDEFGHIJKLMNOPQRST\n"
                            " 3 ..............*++++*  3\n"
                            " 2 ..............+oooo+  2\n"
                            " 1 ..............*+o++*  1\n"
                            "   ABCDEFGHIJKLMNOPQRST\n")) {
    return;
  }

  if (!full_piece_repr_test(__FILE__, __LINE__, poc, 2, 16,
                            "   ABCDEFGHIJKLMNOPQRST\n"
                            " 4 ...............*++++  4\n"
                            " 3 ...............+oooo  3\n"
                            " 2 ...............*+o++  2\n"
                            " 1 ................*+*.  1\n"
                            "   ABCDEFGHIJKLMNOPQRST\n")) {
    return;
  }

  if (!full_piece_repr_test(__FILE__, __LINE__, poc, 2, 15,
                            "   ABCDEFGHIJKLMNOPQRST\n"
                            " 4 ..............*++++*  4\n"
                            " 3 ..............+oooo+  3\n"
                            " 2 ..............*+o++*  2\n"
                            " 1 ...............*+*..  1\n"
                            "   ABCDEFGHIJKLMNOPQRST\n")) {
    return;
  }

  if (!full_piece_repr_test(__FILE__, __LINE__, poc, 19, 16,
                            "   ABCDEFGHIJKLMNOPQRST\n"
                            "20 ...............+oooo 20\n"
                            "19 ...............*+o++ 19\n"
                            "18 ................*+*. 18\n"
                            "   ABCDEFGHIJKLMNOPQRST\n")) {
    return;
  }

  if (!full_piece_repr_test(__FILE__, __LINE__, poc, 19, 15,
                            "   ABCDEFGHIJKLMNOPQRST\n"
                            "20 ..............+oooo+ 20\n"
                            "19 ..............*+o++* 19\n"
                            "18 ...............*+*.. 18\n"
                            "   ABCDEFGHIJKLMNOPQRST\n")) {
    return;
  }

  if (!full_piece_repr_test(__FILE__, __LINE__, poc, 18, 16,
                            "   ABCDEFGHIJKLMNOPQRST\n"
                            "20 ...............*++++ 20\n"
                            "19 ...............+oooo 19\n"
                            "18 ...............*+o++ 18\n"
                            "17 ................*+*. 17\n"
                            "   ABCDEFGHIJKLMNOPQRST\n")) {
    return;
  }

  if (!full_piece_repr_test(__FILE__, __LINE__, poc, 18, 15,
                            "   ABCDEFGHIJKLMNOPQRST\n"
                            "20 ..............*++++* 20\n"
                            "19 ..............+oooo+ 19\n"
                            "18 ..............*+o++* 18\n"
                            "17 ...............*+*.. 17\n"
                            "   ABCDEFGHIJKLMNOPQRST\n")) {
    return;
  }

  printf("Success: %s()\n", __func__);
  global_pass_count++;
}

int main() {
  util::set_tty_mode(false);
  test_location();
  test_bit_board();
  test_piece_orientation_corner();

  if (global_fail_count > 0) {
    int total_count = global_pass_count + global_fail_count;
    printf("Failed %d of %d test%s!\n", global_fail_count, total_count, total_count > 1 ? "s" : "");
  } else {
    printf("All tests passed (%d of %d)!\n", global_pass_count, global_pass_count);
  }
  return global_fail_count ? 1 : 0;
}
