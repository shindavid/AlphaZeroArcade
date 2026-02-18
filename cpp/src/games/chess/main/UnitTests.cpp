#include "games/chess/Game.hpp"
#include "lc0/chess/types.h"
#include "lc0/chess/board.h"
#include "lc0/neural/encoder.h"
#include "util/GTestUtil.hpp"

#include "gtest/gtest.h"

#include <iostream>
#include <sstream>
#include <algorithm>


#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using Game = chess::Game;
using State = Game::State;

std::string convert_to_fen(const std::string& board_str) {
    std::stringstream ss(board_str);
    std::string line;
    std::string fen_body = "";

    // Default tail if none is found in the string.
    // We include a leading space so we can just append it later.
    std::string fen_tail = " w KQkq - 0 1";

    while (std::getline(ss, line)) {
        // 1. Find the board boundaries (the pipes '|')
        size_t first_pipe = line.find('|');
        size_t last_pipe = line.rfind('|');

        // Check if this line is NOT a board rank
        if (first_pipe == std::string::npos || last_pipe == std::string::npos || first_pipe == last_pipe) {

            // It might be the header ("a b c") or the footer ("w KQkq...").
            // Let's trim whitespace to check.
            std::string trimmed = line;
            size_t first_char = trimmed.find_first_not_of(" \t\r\n");
            if (first_char == std::string::npos) continue; // Skip empty lines
            trimmed = trimmed.substr(first_char);

            // Skip the coordinate header
            if (trimmed.find("a b c") != std::string::npos) continue;

            // Heuristic: If it starts with 'w' or 'b', assume it's the FEN tail.
            if (trimmed[0] == 'w' || trimmed[0] == 'b') {
                fen_tail = " " + trimmed; // Overwrite default, ensure leading space
            }
            continue;
        }

        // 2. Process strictly the content *inside* the outer pipes
        std::string content = line.substr(first_pipe + 1, last_pipe - first_pipe - 1);

        // 3. Remove all separator pipes ('|')
        content.erase(std::remove(content.begin(), content.end(), '|'), content.end());

        // 4. Convert the clean string to FEN (handling empty spaces)
        std::string rank_fen = "";
        int empty_count = 0;

        for (char c : content) {
            if (c == ' ') {
                empty_count++;
            } else {
                if (empty_count > 0) {
                    rank_fen += std::to_string(empty_count);
                    empty_count = 0;
                }
                rank_fen += c;
            }
        }
        if (empty_count > 0) {
            rank_fen += std::to_string(empty_count);
        }

        // 5. Append to the full FEN body
        if (!fen_body.empty()) {
            fen_body += "/";
        }
        fen_body += rank_fen;
    }

    // Combine the parsed board with the detected (or default) tail
    return fen_body + fen_tail;
}

lczero::Move UciToMove(const std::string& uci) {
  // 1. Parse the Source and Destination squares
  // Lc0 has a helper Square::Parse(std::string) implied by your encoder.cc
  auto from = lczero::Square::Parse(uci.substr(0, 2));
  auto to = lczero::Square::Parse(uci.substr(2, 2));

  // 2. Handle Promotions (e.g., "a7a8q")
  if (uci.length() == 5) {
    // PieceType::Parse creates a piece from char ('q', 'r', 'b', 'n')
    auto promotion_piece = lczero::PieceType::Parse(uci[4]);
    return lczero::Move::WhitePromotion(from, to, promotion_piece);
  }

  // 3. Handle Regular Moves (and Castling*)
  // *Note: In this context, Lc0 treats a castling string (e.g., "e1g1")
  // simply as a King moving from e1 to g1.
  return lczero::Move::White(from, to);
}

core::action_t UciToAction(const std::string& uci, char side_to_move) {
  lczero::Move move = UciToMove(uci);
  if (side_to_move == 'b') {
    move.Flip();
  }
  return lczero::MoveToNNIndex(move, 0);
}

TEST(StartingPosition, board) {
  State state;
  Game::Rules::init_state(state);

  const std::string board_string =
    "   a b c d e f g h\n"
    " 8|r|n|b|q|k|b|n|r|\n"
    " 7|p|p|p|p|p|p|p|p|\n"
    " 6| | | | | | | | |\n"
    " 5| | | | | | | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2|P|P|P|P|P|P|P|P|\n"
    " 1|R|N|B|Q|K|B|N|R|\n"
    " w KQkq - 0 1\n";

  std::string generated_fen = convert_to_fen(board_string);
  std::string expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

  EXPECT_EQ(generated_fen, expected_fen) << "FEN Converter failed!";

  State::ChessBoard board(generated_fen);
  EXPECT_EQ(board, state.board);
}

TEST(BoardMove, WhitePawnPush_e2e4) {
  State state;
  Game::Rules::init_state(state);
  core::action_t action = UciToAction("e2e4", 'w');
  Game::Rules::apply(state, action);

  const std::string expected_board_str =
    "   a b c d e f g h\n"
    " 8|r|n|b|q|k|b|n|r|\n"
    " 7|p|p|p|p|p|p|p|p|\n"
    " 6| | | | | | | | |\n"
    " 5| | | | | | | | |\n"
    " 4| | | | |P| | | |\n"
    " 3| | | | | | | | |\n"
    " 2|P|P|P|P| |P|P|P|\n"
    " 1|R|N|B|Q|K|B|N|R|\n"
    " b KQkq - 0 1\n";

  std::string generated_fen = convert_to_fen(expected_board_str);
  std::string expected_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
  EXPECT_EQ(generated_fen, expected_fen);
  State::ChessBoard expected_board(expected_fen);

  EXPECT_EQ(state.board, expected_board);
}

TEST(BoardMove, BlackPawnPush_e7e5) {
  State state;
  Game::Rules::init_state(state);
  core::action_t action1 = UciToAction("e2e4", 'w');
  core::action_t action2 = UciToAction("f7f5", 'b');
  Game::Rules::apply(state, action1);
  Game::Rules::apply(state, action2);

  const std::string expected_board_str =
    "   a b c d e f g h\n"
    " 8|r|n|b|q|k|b|n|r|\n"
    " 7|p|p|p|p|p| |p|p|\n"
    " 6| | | | | | | | |\n"
    " 5| | | | | |p| | |\n"
    " 4| | | | |P| | | |\n"
    " 3| | | | | | | | |\n"
    " 2|P|P|P|P| |P|P|P|\n"
    " 1|R|N|B|Q|K|B|N|R|\n"
    " w KQkq - 0 2\n";

  std::string expected_fen = convert_to_fen(expected_board_str);
  State::ChessBoard expected_board(expected_fen);
  EXPECT_EQ(state.board, expected_board);
}

TEST(BoardMove, WhiteCaptures_e4f5) {
  State state;
  Game::Rules::init_state(state);
  core::action_t action1 = UciToAction("e2e4", 'w');
  core::action_t action2 = UciToAction("f7f5", 'b');
  core::action_t action3 = UciToAction("e4f5", 'w');
  Game::Rules::apply(state, action1);
  Game::Rules::apply(state, action2);
  Game::Rules::apply(state, action3);

  const std::string expected_board_str =
    "   a b c d e f g h\n"
    " 8|r|n|b|q|k|b|n|r|\n"
    " 7|p|p|p|p|p| |p|p|\n"
    " 6| | | | | | | | |\n"
    " 5| | | | | |P| | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2|P|P|P|P| |P|P|P|\n"
    " 1|R|N|B|Q|K|B|N|R|\n"
    " b KQkq - 0 2\n";

  std::string expected_fen = convert_to_fen(expected_board_str);
  State::ChessBoard expected_board(expected_fen);
  EXPECT_EQ(state.board, expected_board);
}

TEST(BoardMove, EnPassant_e7e5) {
  State state;
  Game::Rules::init_state(state);
  core::action_t action1 = UciToAction("e2e4", 'w');
  core::action_t action2 = UciToAction("f7f5", 'b');
  core::action_t action3 = UciToAction("e4f5", 'w');
  core::action_t action4 = UciToAction("e7e5", 'b');

  Game::Rules::apply(state, action1);
  Game::Rules::apply(state, action2);
  Game::Rules::apply(state, action3);
  Game::Rules::apply(state, action4);

  const std::string expected_board_str =
    "   a b c d e f g h\n"
    " 8|r|n|b|q|k|b|n|r|\n"
    " 7|p|p|p|p| | |p|p|\n"
    " 6| | | | | | | | |\n"
    " 5| | | | |p|P| | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2|P|P|P|P| |P|P|P|\n"
    " 1|R|N|B|Q|K|B|N|R|\n"
    " w KQkq e6 0 2\n";

  std::string expected_fen = convert_to_fen(expected_board_str);
  State::ChessBoard expected_board(expected_fen);
  EXPECT_EQ(state.board, expected_board);
}

TEST(IsTerminal, Checkmate) {
  State state;
  const std::string board_str =
    "   a b c d e f g h\n"
    " 8| | | | | | | |k|\n"
    " 7| | | | | | |Q| |\n"
    " 6| | | | | |K| | |\n"
    " 5| | | | | | | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2| | | | | | | | |\n"
    " 1| | | | | | | | |\n"
    " b - - 0 1\n";

  std::string fen = convert_to_fen(board_str);
  State::ChessBoard board(fen);
  state.board = board;

  Game::GameResults::Tensor outcome;
  bool is_terminal = Game::Rules::is_terminal(state, 0, -1, outcome);

  EXPECT_TRUE(is_terminal);

  EXPECT_EQ(outcome(0), 1);
  EXPECT_EQ(outcome(1), 0);
  EXPECT_EQ(outcome(2), 0);
}

TEST(IsTerminal, Stalemate) {
  State state;
  const std::string board_str =
    "   a b c d e f g h\n"
    " 8| | | | | | | |k|\n"
    " 7| | | | |K| | | |\n"
    " 6| | | | | | |Q| |\n"
    " 5| | | | | | | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2| | | | | | | | |\n"
    " 1| | | | | | | | |\n"
    " b - - 0 1\n";

  std::string fen = convert_to_fen(board_str);
  State::ChessBoard board(fen);
  state.board = board;

  Game::GameResults::Tensor outcome;
  bool is_terminal = Game::Rules::is_terminal(state, 0, -1, outcome);

  EXPECT_TRUE(is_terminal);

  EXPECT_EQ(outcome(0), 0);
  EXPECT_EQ(outcome(1), 0);
  EXPECT_EQ(outcome(2), 1); // Expect Draw = 1
}

TEST(IsTerminal, ThreeFoldRepetition) {
  State state;
  const std::string board_str =
    "   a b c d e f g h\n"
    " 8|r| | | |k| | | |\n"
    " 7| | | | | | | | |\n"
    " 6| | | | | | | | |\n"
    " 5| | | | | | | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2| | | | | | | | |\n"
    " 1|R| | | |K| | | |\n"
    " w - - 0 1\n";
  std::string fen = convert_to_fen(board_str);
  State::ChessBoard board(fen);
  state.board = board;

  for (int i = 0; i < 3; ++i) {
    core::action_t action1 = UciToAction("a1a2", 'w');
    core::action_t action2 = UciToAction("a8a7", 'b');
    core::action_t action3 = UciToAction("a2a1", 'w');
    core::action_t action4 = UciToAction("a7a8", 'b');

    Game::Rules::apply(state, action1);
    Game::Rules::apply(state, action2);
    Game::Rules::apply(state, action3);
    Game::Rules::apply(state, action4);
  }
  Game::GameResults::Tensor outcome;
  bool is_terminal = Game::Rules::is_terminal(state, 0, -1, outcome);
  EXPECT_TRUE(is_terminal);
  EXPECT_EQ(outcome(0), 0);
  EXPECT_EQ(outcome(1), 0);
  EXPECT_EQ(outcome(2), 1);
}

TEST(Symmetry, square_flip_vertical) {
  lczero::Square sq = lczero::Square::Parse("e4");
  sq.flip_vertical();
  EXPECT_EQ(sq.ToString(), "e5");
}

TEST(Symmetry, square_mirror_horizontal) {
  lczero::Square sq = lczero::Square::Parse("e4");
  sq.mirror_horizontal();
  EXPECT_EQ(sq.ToString(), "d4");
}

TEST(Symmetry, square_flip_main_diag) {
  lczero::Square sq = lczero::Square::Parse("e4");
  sq.flip_main_diag();
  EXPECT_EQ(sq.ToString(), "d5");
}

TEST(Symmetry, square_flip_anti_diag) {
  lczero::Square sq = lczero::Square::Parse("e4");
  sq.flip_anti_diag();
  EXPECT_EQ(sq.ToString(), "e4");
}

TEST(Symmetry, square_rot90_clockwise) {
  lczero::Square sq = lczero::Square::Parse("e4");
  sq.rot90_clockwise();
  EXPECT_EQ(sq.ToString(), "d4");
}

TEST(Symmetry, square_rot180) {
  lczero::Square sq = lczero::Square::Parse("e4");
  sq.rot180();
  EXPECT_EQ(sq.ToString(), "d5");
}

TEST(Symmetry, square_rot270_clockwise) {
  lczero::Square sq = lczero::Square::Parse("e4");
  sq.rot270_clockwise();
  EXPECT_EQ(sq.ToString(), "e5");
}

TEST(Symmetry, board_flip_vertical) {
  const std::string board_str =
    "   a b c d e f g h\n"
    " 8|r|n|b|q|k|b|n|r|\n"
    " 7|p|p|p|p|p|p|p|p|\n"
    " 6| | | | | | | | |\n"
    " 5| | | | | | | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2|P|P|P|P|P|P|P|P|\n"
    " 1|R|N|B|Q|K|B|N|R|\n"
    " w - - 0 1\n";

  std::string fen = convert_to_fen(board_str);
  State::ChessBoard board(fen);
  board.flip_vertical();

  const std::string expected_board_str =
    "   a b c d e f g h\n"
    " 8|R|N|B|Q|K|B|N|R|\n"
    " 7|P|P|P|P|P|P|P|P|\n"
    " 6| | | | | | | | |\n"
    " 5| | | | | | | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2|p|p|p|p|p|p|p|p|\n"
    " 1|r|n|b|q|k|b|n|r|\n"
    " w - - 0 1\n";

  std::string expected_fen = convert_to_fen(expected_board_str);
  State::ChessBoard expected_board(expected_fen);
  EXPECT_EQ(board, expected_board);
}

TEST(Symmetry, board_mirror_horizontal) {
  const std::string board_str =
    "   a b c d e f g h\n"
    " 8|r|n|b|q|k|b|n|r|\n"
    " 7|p|p|p|p|p|p|p|p|\n"
    " 6| | | | | | | | |\n"
    " 5| | | | | | | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2|P|P|P|P|P|P|P|P|\n"
    " 1|R|N|B|Q|K|B|N|R|\n"
    " w - - 0 1\n";

  std::string fen = convert_to_fen(board_str);
  State::ChessBoard board(fen);
  board.mirror_horizontal();

  const std::string expected_board_str =
    "   a b c d e f g h\n"
    " 8|r|n|b|k|q|b|n|r|\n"
    " 7|p|p|p|p|p|p|p|p|\n"
    " 6| | | | | | | | |\n"
    " 5| | | | | | | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2|P|P|P|P|P|P|P|P|\n"
    " 1|R|N|B|K|Q|B|N|R|\n"
    " w - - 0 1\n";

  std::string expected_fen = convert_to_fen(expected_board_str);
  State::ChessBoard expected_board(expected_fen);
  EXPECT_EQ(board, expected_board);
}

TEST(Symmetry, board_rot90_clockwise) {
  const std::string board_str =
    "   a b c d e f g h\n"
    " 8|r|n|b|q|k|b|n|r|\n"
    " 7| | | | | | | | |\n"
    " 6| | | | | | | | |\n"
    " 5| | | | | | | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2| | | | | | | | |\n"
    " 1|R|N|B|Q|K|B|N|R|\n"
    " w - - 0 1\n";

  std::string fen = convert_to_fen(board_str);
  State::ChessBoard board(fen);
  board.rot90_clockwise();

  const std::string expected_board_str =
    "   a b c d e f g h\n"
    " 8|R| | | | | | |r|\n"
    " 7|N| | | | | | |n|\n"
    " 6|B| | | | | | |b|\n"
    " 5|Q| | | | | | |q|\n"
    " 4|K| | | | | | |k|\n"
    " 3|B| | | | | | |b|\n"
    " 2|N| | | | | | |n|\n"
    " 1|R| | | | | | |r|\n"
    " w - - 0 1\n";

  std::string expected_fen = convert_to_fen(expected_board_str);
  State::ChessBoard expected_board(expected_fen);
  EXPECT_EQ(board, expected_board);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
