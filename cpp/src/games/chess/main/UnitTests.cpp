#include "games/chess/Game.hpp"
#include "games/chess/MoveEncoder.hpp"
#include "util/GTestUtil.hpp"

#include "gtest/gtest.h"

#include <sstream>
#include <algorithm>
#include <string>



#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using Game = a0achess::Game;
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
  return a0achess::move_to_nn_idx(board, move);
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
  EXPECT_EQ(state.getFen(), expected_fen);
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
  EXPECT_EQ(state.getFen(), expected_fen);
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
  EXPECT_EQ(state.getFen(), expected_fen);
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
  EXPECT_EQ(state.getFen(), expected_fen);
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
  EXPECT_EQ(state.getFen(), expected_fen);
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
  State state(fen);

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
  State state(fen);

  Game::GameResults::Tensor outcome;
  bool is_terminal = Game::Rules::is_terminal(state, 0, -1, outcome);

  EXPECT_TRUE(is_terminal);

  EXPECT_EQ(outcome(0), 0);
  EXPECT_EQ(outcome(1), 0);
  EXPECT_EQ(outcome(2), 1); // Expect Draw = 1
}

TEST(IsTerminal, ThreeFoldRepetition) {
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
  State state(fen);

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

TEST(CompactState, StartingPosition) {
  State state;
  Game::Rules::init_state(state);

  auto compact = state.to_compact_state();

  // All pieces for each side
  EXPECT_EQ(compact.all_pieces[a0achess::kWhite], state.us(Color::WHITE));
  EXPECT_EQ(compact.all_pieces[a0achess::kBlack], state.us(Color::BLACK));

  // Orthogonal movers: rooks + queens
  chess::Bitboard expected_ortho = state.pieces(chess::PieceType::ROOK) | state.pieces(chess::PieceType::QUEEN);
  EXPECT_EQ(compact.orthogonal_movers, expected_ortho);

  // Diagonal movers: bishops + queens
  chess::Bitboard expected_diag = state.pieces(chess::PieceType::BISHOP) | state.pieces(chess::PieceType::QUEEN);
  EXPECT_EQ(compact.diagonal_movers, expected_diag);

  // Pawns (no en passant, so just the raw pawn bitboard)
  EXPECT_EQ(compact.pawns, state.pieces(chess::PieceType::PAWN));

  // Kings
  EXPECT_EQ(compact.kings[a0achess::kWhite], static_cast<a0achess::Square>(Square::underlying::SQ_E1));
  EXPECT_EQ(compact.kings[a0achess::kBlack], static_cast<a0achess::Square>(Square::underlying::SQ_E8));

  // Castling: all four rights
  EXPECT_EQ(compact.castling_rights, 0b1111);

  // Side to move
  EXPECT_EQ(compact.cur_player, a0achess::kWhite);

  // Half move clock
  EXPECT_EQ(compact.half_move_clock, 0);
}

TEST(CompactState, AfterE4) {
  State state;
  Game::Rules::init_state(state);

  auto e4 = chess::uci::uciToMove(state, "e2e4");
  state.makeMove(e4);

  auto compact = state.to_compact_state();

  EXPECT_EQ(compact.cur_player, a0achess::kBlack);
  EXPECT_EQ(compact.half_move_clock, 0);
  EXPECT_EQ(compact.castling_rights, 0b1111);

  // No en passant encoding because no enemy pawn can capture
  // (ep square may or may not be set depending on EXACT, but no black pawn on d5/f5)
  // Pawns should just be the raw pawn bitboard
  EXPECT_EQ(compact.all_pieces[a0achess::kWhite], state.us(Color::WHITE));
  EXPECT_EQ(compact.all_pieces[a0achess::kBlack], state.us(Color::BLACK));
}

TEST(CompactState, EnPassantEncoding) {
  // Position where en passant is available
  // White pawn on e5, black plays d7d5 -> ep square is d6
  State state;
  const std::string board_str =
    "   a b c d e f g h\n"
    " 8|r|n|b|q|k|b|n|r|\n"
    " 7|p|p|p| |p|p|p|p|\n"
    " 6| | | | | | | | |\n"
    " 5| | | |p|P| | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2|P|P|P|P| |P|P|P|\n"
    " 1|R|N|B|Q|K|B|N|R|\n"
    " w KQkq d6 0 3\n";

  std::string fen = convert_to_fen(board_str);
  state.setFen(fen);

  auto compact = state.to_compact_state();

  uint64_t expected_white_pawns =
    1 << a0achess::Square::kA2 | 1 << a0achess::Square::kB2 | 1 << a0achess::Square::kC2 |
    1 << a0achess::Square::kD2 | 1 << a0achess::Square::kF2 | 1 << a0achess::Square::kG2 |
    1 << a0achess::Square::kH2 | 1 << a0achess::Square::kE5;

  uint64_t expected_black_pawns =
    1 << a0achess::Square::kA7 | 1 << a0achess::Square::kB7 | 1 << a0achess::Square::kC7 |
    1 << a0achess::Square::kE7 | 1 << a0achess::Square::kF7 | 1 << a0achess::Square::kG7 |
    1 << a0achess::Square::kH7 | 1 << a0achess::Square::kD5;

  uint64_t expected_ep_flag = 1 << a0achess::Square::kD8;  // en passant flag for d6

  EXPECT_EQ(compact.pawns, expected_black_pawns | expected_white_pawns | expected_ep_flag);
  EXPECT_EQ(compact.get(chess::PieceType::PAWN, a0achess::kWhite), expected_white_pawns);
  EXPECT_EQ(compact.get(chess::PieceType::PAWN, a0achess::kBlack), expected_black_pawns);
  EXPECT_EQ(compact.get_en_passant(), expected_ep_flag)
    << std::format("compact ep: {:#018x}\nexpected ep flag {:#018x}",
                   compact.get_en_passant().getBits(), expected_ep_flag);
}

TEST(CompactState, NoCastlingRights) {
  State state;
  const std::string board_str =
    "   a b c d e f g h\n"
    " 8|r| | | |k| | |r|\n"
    " 7|p|p|p|p|p|p|p|p|\n"
    " 6| | | | | | | | |\n"
    " 5| | | | | | | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2|P|P|P|P|P|P|P|P|\n"
    " 1|R|N|B|Q|K|B|N|R|\n"
    " w - - 0 1\n";

  std::string fen = convert_to_fen(board_str);
  state.setFen(fen);

  auto compact = state.to_compact_state();

  EXPECT_EQ(compact.castling_rights, 0);
}

TEST(CompactState, PartialCastlingRights) {
  State state;
  std::string board_str =
    "   a b c d e f g h\n"
    " 8|r| | | |k| | |r|\n"
    " 7|p|p|p|p|p|p|p|p|\n"
    " 6| | | | | | | | |\n"
    " 5| | | | | | | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2|P|P|P|P|P|P|P|P|\n"
    " 1|R|N|B|Q|K|B|N|R|\n"
    " w Kq - 0 1\n";
  std::string fen = convert_to_fen(board_str);
  state.setFen(fen);

  auto compact = state.to_compact_state();

  uint8_t expected = (1 << a0achess::CastlingRightBit::kWhiteKingSide) |
                     (1 << a0achess::CastlingRightBit::kBlackQueenSide);
  EXPECT_EQ(compact.castling_rights, expected);
}

TEST(CompactState, KingsPosition) {
  State state;
  std::string board_str =
    "   a b c d e f g h\n"
    " 8| | | | | | | | |\n"
    " 7| | | | | | | | |\n"
    " 6| | | | | | | | |\n"
    " 5| | | | |k| | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2| | | | | | | | |\n"
    " 1|K| | | | | | | |\n"
    " w - - 0 1\n";

  std::string fen = convert_to_fen(board_str);
  state.setFen(fen);

  auto compact = state.to_compact_state();

  EXPECT_EQ(compact.kings[a0achess::kWhite], static_cast<a0achess::Square>(Square("a1").index()));
  EXPECT_EQ(compact.kings[a0achess::kBlack], static_cast<a0achess::Square>(Square("e5").index()));
}

TEST(CompactState, HalfMoveClock) {
  State state;

  std::string board_str =
    "   a b c d e f g h\n"
    " 8|r|n|b|q|k|b|n|r|\n"
    " 7|p|p|p|p|p|p|p|p|\n"
    " 6| | | | | | | | |\n"
    " 5| | | | | | | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2|P|P|P|P|P|P|P|P|\n"
    " 1|R|N|B|Q|K|B|N|R|\n"
    " w KQkq - 42 1\n";

  std::string fen = convert_to_fen(board_str);
  state.setFen(fen);

  auto compact = state.to_compact_state();

  EXPECT_EQ(compact.half_move_clock, 42);
}

TEST(CompactState, PieceBitboardRecovery) {
  // Verify we can recover individual piece types from the compact representation
  State state;
  const std::string board_str =
    "   a b c d e f g h\n"
    " 8|r|n|b|q|k|b|n|r|\n"
    " 7|p|p|p| |p|p|p|p|\n"
    " 6| | | | | | | | |\n"
    " 5| | | |p|P| | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2|P|P|P|P| |P|P|P|\n"
    " 1|R|N|B|Q|K|B|N|R|\n"
    " w KQkq d6 0 3\n";

  std::string fen = convert_to_fen(board_str);
  state.setFen(fen);

  auto compact = state.to_compact_state();

  EXPECT_EQ(compact.get(chess::PieceType::PAWN, a0achess::kWhite),
            state.pieces(chess::PieceType::PAWN, chess::Color::WHITE));
  EXPECT_EQ(compact.get(chess::PieceType::PAWN, a0achess::kBlack),
            state.pieces(chess::PieceType::PAWN, chess::Color::BLACK));

  EXPECT_EQ(compact.get(chess::PieceType::KNIGHT, a0achess::kWhite),
            state.pieces(chess::PieceType::KNIGHT, chess::Color::WHITE));
  EXPECT_EQ(compact.get(chess::PieceType::KNIGHT, a0achess::kBlack),
            state.pieces(chess::PieceType::KNIGHT, chess::Color::BLACK));

  EXPECT_EQ(compact.get(chess::PieceType::BISHOP, a0achess::kWhite),
            state.pieces(chess::PieceType::BISHOP, chess::Color::WHITE));
  EXPECT_EQ(compact.get(chess::PieceType::BISHOP, a0achess::kBlack),
            state.pieces(chess::PieceType::BISHOP, chess::Color::BLACK));

  EXPECT_EQ(compact.get(chess::PieceType::ROOK, a0achess::kWhite),
            state.pieces(chess::PieceType::ROOK, chess::Color::WHITE));
  EXPECT_EQ(compact.get(chess::PieceType::ROOK, a0achess::kBlack),
            state.pieces(chess::PieceType::ROOK, chess::Color::BLACK));

  EXPECT_EQ(compact.get(chess::PieceType::QUEEN, a0achess::kWhite),
            state.pieces(chess::PieceType::QUEEN, chess::Color::WHITE));
  EXPECT_EQ(compact.get(chess::PieceType::QUEEN, a0achess::kBlack),
            state.pieces(chess::PieceType::QUEEN, chess::Color::BLACK));

  EXPECT_EQ(compact.get(chess::PieceType::KING, a0achess::kWhite),
            state.pieces(chess::PieceType::KING, chess::Color::WHITE));
  EXPECT_EQ(compact.get(chess::PieceType::KING, a0achess::kBlack),
            state.pieces(chess::PieceType::KING, chess::Color::BLACK));

}

int main(int argc, char** argv) {
  return launch_gtest(argc, argv);
}
