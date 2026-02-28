#include "games/chess/Game.hpp"
#include "games/chess/MoveEncoder.hpp"
#include "util/GTestUtil.hpp"

#include "gtest/gtest.h"

#include <sstream>
#include <algorithm>



#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using Game = chess::Game;
using State = Game::State;
using Move = chess::Move;
using Square = chess::Square;
using Board = chess::Board;
using Color = chess::Color;
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

core::action_t UciToAction(const Board& board,const std::string& uci) {
  Move move = chess::uci::uciToMove(board, uci);
  return chess::move_to_nn_idx(board, move);
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
  EXPECT_EQ(state.fen(), expected_fen);
}

TEST(BoardMove, WhitePawnPush_e2e4) {
  State state;
  Game::Rules::init_state(state);
  core::action_t action = state.action_from_uci("e2e4");
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

  std::string expected_fen = convert_to_fen(expected_board_str);
  EXPECT_EQ(state.fen(), expected_fen);
}

TEST(BoardMove, BlackPawnPush_e7e5) {
  State state;
  Game::Rules::init_state(state);
  core::action_t action1 = state.action_from_uci("e2e4");
  Game::Rules::apply(state, action1);
  core::action_t action2 = state.action_from_uci("f7f5");
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
  EXPECT_EQ(state.fen(), expected_fen);
}

TEST(BoardMove, WhiteCaptures_e4f5) {
  State state;
  Game::Rules::init_state(state);
  core::action_t action1 = state.action_from_uci("e2e4");
  Game::Rules::apply(state, action1);

  core::action_t action2 = state.action_from_uci("f7f5");
  Game::Rules::apply(state, action2);

  core::action_t action3 = state.action_from_uci("e4f5");
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
  EXPECT_EQ(state.fen(), expected_fen);
}

TEST(BoardMove, EnPassant_e7e5) {
  State state;
  Game::Rules::init_state(state);
  core::action_t action1 = state.action_from_uci("e2e4");
  Game::Rules::apply(state, action1);

  core::action_t action2 = state.action_from_uci("f7f5");
  Game::Rules::apply(state, action2);

  core::action_t action3 = state.action_from_uci("e4f5");
  Game::Rules::apply(state, action3);

  core::action_t action4 = state.action_from_uci("e7e5");
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
    " w KQkq e6 0 3\n";

  std::string expected_fen = convert_to_fen(expected_board_str);
  EXPECT_EQ(state.fen(), expected_fen);
}

TEST(IsTerminal, Checkmate) {
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
  Board board(fen);
  State state(board);

  Game::GameResults::Tensor outcome;
  bool is_terminal = Game::Rules::is_terminal(state, 0, -1, outcome);

  EXPECT_TRUE(is_terminal);

  EXPECT_EQ(outcome(0), 1);
  EXPECT_EQ(outcome(1), 0);
  EXPECT_EQ(outcome(2), 0);
}

TEST(IsTerminal, Stalemate) {
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
  Board board(fen);
  State state(board);

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
  Board board(fen);
  state = State(board);

  for (int i = 0; i < 3; ++i) {
    core::action_t action1 = state.action_from_uci("a1a2");
    Game::Rules::apply(state, action1);

    core::action_t action2 = state.action_from_uci("a8a7");
    Game::Rules::apply(state, action2);

    core::action_t action3 = state.action_from_uci("a2a1");
    Game::Rules::apply(state, action3);

    core::action_t action4 = state.action_from_uci("a7a8");
    Game::Rules::apply(state, action4);
  }

  Game::GameResults::Tensor outcome;
  bool is_terminal = Game::Rules::is_terminal(state, 0, -1, outcome);
  EXPECT_TRUE(is_terminal);
  EXPECT_EQ(outcome(0), 0);
  EXPECT_EQ(outcome(1), 0);
  EXPECT_EQ(outcome(2), 1);
}

int main(int argc, char** argv) {
  return launch_gtest(argc, argv);
}
