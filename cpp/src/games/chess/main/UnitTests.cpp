#include "games/chess/Game.hpp"
#include "util/GTestUtil.hpp"
#include <sstream>
#include <vector>
#include <algorithm>

#include "gtest/gtest.h"

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using Game = chess::Game;
using State = Game::State;

// --- Helper Function: Converts Visual Board to FEN ---
std::string convert_to_fen(const std::string& board_str) {
    std::stringstream ss(board_str);
    std::string line;
    std::string fen_body = "";

    while (std::getline(ss, line)) {
        // 1. Find the board boundaries (the first and last pipe '|')
        size_t first_pipe = line.find('|');
        size_t last_pipe = line.rfind('|');

        // If no pipes, this isn't a board rank (it's likely the "a b c" or empty line)
        if (first_pipe == std::string::npos || last_pipe == std::string::npos || first_pipe == last_pipe) {
            continue;
        }

        // 2. Extract strictly the content *inside* the outer pipes
        // Example: " 8|r|n|b|..." -> "r|n|b|..."
        std::string content = line.substr(first_pipe + 1, last_pipe - first_pipe - 1);

        // 3. Remove all separator pipes ('|') from the content
        // Example: "r|n|b|..." -> "rnb..."
        // Example: " | | |..." -> "        " (8 spaces)
        content.erase(std::remove(content.begin(), content.end(), '|'), content.end());

        // 4. Convert the clean 8-char string to FEN (handling empty spaces)
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

        // 5. Append to the full FEN string
        if (!fen_body.empty()) {
            fen_body += "/";
        }
        fen_body += rank_fen;
    }

    // Default: White to move, Castling enabled, No En Passant, Clocks 0 1
    return fen_body + " w KQkq - 0 1";
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
    " 1|R|N|B|Q|K|B|N|R|\n";

  std::string generated_fen = convert_to_fen(board_string);
  std::string expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

  // This check will now pass
  EXPECT_EQ(generated_fen, expected_fen) << "FEN Converter failed!";

  // NOTE: Depending on your specific API, you might need Position::FromFen
  // If State::ChessBoard takes a FEN string directly:
  State::ChessBoard board(generated_fen);
  EXPECT_EQ(board, state.board);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
