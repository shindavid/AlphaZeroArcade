#include "games/chess/Game.hpp"
#include "games/chess/InputFrame.hpp"
#include "games/chess/Move.hpp"
#include "games/chess/SyzygyTable.hpp"
#include "gtest/gtest.h"
#include "util/GTestUtil.hpp"

#include <chess-library/include/chess.hpp>

#include <algorithm>
#include <sstream>
#include <string>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using Game = a0achess::Game;
using State = Game::State;
using IO = Game::IO;
using InputFrame = a0achess::InputFrame;
using Move = a0achess::Move;
using MoveList = a0achess::MoveList;
using SyzygyTable = a0achess::SyzygyTable;
using Square = chess::Square;
using Board = chess::Board;
using Color = chess::Color;
using State = Game::State;
using Rules = Game::Rules;

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
    if (first_pipe == std::string::npos || last_pipe == std::string::npos ||
        first_pipe == last_pipe) {
      // It might be the header ("a b c") or the footer ("w KQkq...").
      // Let's trim whitespace to check.
      std::string trimmed = line;
      size_t first_char = trimmed.find_first_not_of(" \t\r\n");
      if (first_char == std::string::npos) continue;  // Skip empty lines
      trimmed = trimmed.substr(first_char);

      // Skip the coordinate header
      if (trimmed.find("a b c") != std::string::npos) continue;

      // Heuristic: If it starts with 'w' or 'b', assume it's the FEN tail.
      if (trimmed[0] == 'w' || trimmed[0] == 'b') {
        fen_tail = " " + trimmed;  // Overwrite default, ensure leading space
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

TEST(Analyze, FromInitState) {
  State state;
  Rules::init_state(state);

  MoveList valid_moves = Rules::analyze(state).valid_moves();

  std::set<Move> expected_moves;
  // Pawns
  expected_moves.insert(Move(chess::Move::make(Square::SQ_A2, Square::SQ_A3), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_A2, Square::SQ_A4), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_B2, Square::SQ_B3), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_B2, Square::SQ_B4), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_C2, Square::SQ_C3), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_C2, Square::SQ_C4), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_D2, Square::SQ_D3), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_D2, Square::SQ_D4), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_E2, Square::SQ_E3), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_E2, Square::SQ_E4), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_F2, Square::SQ_F3), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_F2, Square::SQ_F4), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_G2, Square::SQ_G3), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_G2, Square::SQ_G4), a0achess::kWhiteToMove));

  // Knights
  expected_moves.insert(Move(chess::Move::make(Square::SQ_H2, Square::SQ_H3), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_H2, Square::SQ_H4), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_B1, Square::SQ_A3), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_B1, Square::SQ_C3), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_G1, Square::SQ_F3), a0achess::kWhiteToMove));
  expected_moves.insert(Move(chess::Move::make(Square::SQ_G1, Square::SQ_H3), a0achess::kWhiteToMove));

  EXPECT_EQ(valid_moves.count(), (int)expected_moves.size());
  for (Move move : valid_moves) {
    EXPECT_TRUE(expected_moves.contains(move)) << "Unexpected move: " << IO::move_to_str(move);
  }
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
  Move move = {chess::Move::make(Square::SQ_E2, Square::SQ_E4), a0achess::kWhiteToMove};
  Game::Rules::apply(state, move);

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
  Move move1 = {chess::Move::make(Square::SQ_E2, Square::SQ_E4), a0achess::kWhiteToMove};
  Game::Rules::apply(state, move1);
  Move move2 = {chess::Move::make(Square::SQ_F7, Square::SQ_F5), a0achess::kBlackToMove};
  Game::Rules::apply(state, move2);

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
  Move move1 = {chess::Move::make(Square::SQ_E2, Square::SQ_E4), a0achess::kWhiteToMove};
  Game::Rules::apply(state, move1);

  Move move2 = {chess::Move::make(Square::SQ_F7, Square::SQ_F5), a0achess::kBlackToMove};
  Game::Rules::apply(state, move2);

  Move move3 = {chess::Move::make(Square::SQ_E4, Square::SQ_F5), a0achess::kWhiteToMove};
  Game::Rules::apply(state, move3);

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
  Move move1 = {chess::Move::make(Square::SQ_E2, Square::SQ_E4), a0achess::kWhiteToMove};
  Game::Rules::apply(state, move1);

  Move move2 = {chess::Move::make(Square::SQ_F7, Square::SQ_F5), a0achess::kBlackToMove};
  Game::Rules::apply(state, move2);

  Move move3 = {chess::Move::make(Square::SQ_E4, Square::SQ_F5), a0achess::kWhiteToMove};
  Game::Rules::apply(state, move3);

  Move move4 = {chess::Move::make(Square::SQ_E7, Square::SQ_E5), a0achess::kBlackToMove};
  Game::Rules::apply(state, move4);

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

  auto rules_result = Game::Rules::analyze(state);
  bool is_terminal = rules_result.is_terminal();
  auto outcome = rules_result.outcome();

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

  auto rules_result = Game::Rules::analyze(state);
  bool is_terminal = rules_result.is_terminal();
  auto outcome = rules_result.outcome();

  EXPECT_TRUE(is_terminal);

  EXPECT_EQ(outcome(0), 0);
  EXPECT_EQ(outcome(1), 0);
  EXPECT_EQ(outcome(2), 1);  // Expect Draw = 1
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
    Move move1 = {chess::Move::make(Square::SQ_A1, Square::SQ_A2), a0achess::kWhiteToMove};
    Game::Rules::apply(state, move1);

    Move move2 = {chess::Move::make(Square::SQ_A8, Square::SQ_A7), a0achess::kBlackToMove};
    Game::Rules::apply(state, move2);

    Move move3 = {chess::Move::make(Square::SQ_A2, Square::SQ_A1), a0achess::kWhiteToMove};
    Game::Rules::apply(state, move3);

    Move move4 = {chess::Move::make(Square::SQ_A7, Square::SQ_A8), a0achess::kBlackToMove};
    Game::Rules::apply(state, move4);
  }

  auto rules_result = Game::Rules::analyze(state);
  bool is_terminal = rules_result.is_terminal();
  auto outcome = rules_result.outcome();

  EXPECT_TRUE(is_terminal);
  EXPECT_EQ(outcome(0), 0);
  EXPECT_EQ(outcome(1), 0);
  EXPECT_EQ(outcome(2), 1);
}

TEST(InputFrame, StartingPosition) {
  State state;
  Game::Rules::init_state(state);

  InputFrame frame(state);

  // All pieces for each side
  EXPECT_EQ(frame.all_pieces[a0achess::kWhite], state.us(Color::WHITE));
  EXPECT_EQ(frame.all_pieces[a0achess::kBlack], state.us(Color::BLACK));

  // Orthogonal movers: rooks + queens
  chess::Bitboard expected_ortho =
    state.pieces(chess::PieceType::ROOK) | state.pieces(chess::PieceType::QUEEN);
  EXPECT_EQ(frame.orthogonal_movers, expected_ortho);

  // Diagonal movers: bishops + queens
  chess::Bitboard expected_diag =
    state.pieces(chess::PieceType::BISHOP) | state.pieces(chess::PieceType::QUEEN);
  EXPECT_EQ(frame.diagonal_movers, expected_diag);

  // Pawns (no en passant, so just the raw pawn bitboard)
  EXPECT_EQ(frame.pawns, state.pieces(chess::PieceType::PAWN));

  // Kings
  EXPECT_EQ(frame.kings[a0achess::kWhite],
            static_cast<a0achess::Square>(Square::underlying::SQ_E1));
  EXPECT_EQ(frame.kings[a0achess::kBlack],
            static_cast<a0achess::Square>(Square::underlying::SQ_E8));

  // Castling: all four rights
  EXPECT_EQ(frame.castling_rights, 0b1111);

  // Side to move
  EXPECT_EQ(frame.cur_player, a0achess::kWhite);

  // Half move clock
  EXPECT_EQ(frame.half_move_clock, 0);
}

TEST(InputFrame, AfterE4) {
  State state;
  Game::Rules::init_state(state);

  auto e4 = chess::uci::uciToMove(state, "e2e4");
  state.makeMove(e4);

  InputFrame frame(state);

  EXPECT_EQ(frame.cur_player, a0achess::kBlack);
  EXPECT_EQ(frame.half_move_clock, 0);
  EXPECT_EQ(frame.castling_rights, 0b1111);

  // No en passant encoding because no enemy pawn can capture
  // (ep square may or may not be set depending on EXACT, but no black pawn on d5/f5)
  // Pawns should just be the raw pawn bitboard
  EXPECT_EQ(frame.all_pieces[a0achess::kWhite], state.us(Color::WHITE));
  EXPECT_EQ(frame.all_pieces[a0achess::kBlack], state.us(Color::BLACK));
}

TEST(InputFrame, EnPassantEncoding) {
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

  InputFrame frame(state);

  uint64_t expected_white_pawns = 1 << a0achess::Square::kA2 | 1 << a0achess::Square::kB2 |
                                  1 << a0achess::Square::kC2 | 1 << a0achess::Square::kD2 |
                                  1 << a0achess::Square::kF2 | 1 << a0achess::Square::kG2 |
                                  1 << a0achess::Square::kH2 | 1 << a0achess::Square::kE5;

  uint64_t expected_black_pawns = 1 << a0achess::Square::kA7 | 1 << a0achess::Square::kB7 |
                                  1 << a0achess::Square::kC7 | 1 << a0achess::Square::kE7 |
                                  1 << a0achess::Square::kF7 | 1 << a0achess::Square::kG7 |
                                  1 << a0achess::Square::kH7 | 1 << a0achess::Square::kD5;

  uint64_t expected_ep_flag = 1 << a0achess::Square::kD8;  // en passant flag for d6

  EXPECT_EQ(frame.pawns, expected_black_pawns | expected_white_pawns | expected_ep_flag);
  EXPECT_EQ(frame.get(chess::PieceType::PAWN, a0achess::kWhite), expected_white_pawns);
  EXPECT_EQ(frame.get(chess::PieceType::PAWN, a0achess::kBlack), expected_black_pawns);
  EXPECT_EQ(frame.get_en_passant(), expected_ep_flag)
    << std::format("frame ep: {:#018x}\nexpected ep flag {:#018x}",
                   frame.get_en_passant().getBits(), expected_ep_flag);
}

TEST(InputFrame, NoCastlingRights) {
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

  InputFrame frame(state);

  EXPECT_EQ(frame.castling_rights, 0);
}

TEST(InputFrame, PartialCastlingRights) {
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

  InputFrame frame(state);

  uint8_t expected = (1 << a0achess::CastlingRightBit::kWhiteKingSide) |
                     (1 << a0achess::CastlingRightBit::kBlackQueenSide);
  EXPECT_EQ(frame.castling_rights, expected);
}

TEST(InputFrame, KingsPosition) {
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

  InputFrame frame(state);

  EXPECT_EQ(frame.kings[a0achess::kWhite], static_cast<a0achess::Square>(Square("a1").index()));
  EXPECT_EQ(frame.kings[a0achess::kBlack], static_cast<a0achess::Square>(Square("e5").index()));
}

TEST(InputFrame, HalfMoveClock) {
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

  InputFrame frame(state);

  EXPECT_EQ(frame.half_move_clock, 42);
}

TEST(InputFrame, PieceBitboardRecovery) {
  // Verify we can recover individual piece types from the frame representation
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

  InputFrame frame(state);

  EXPECT_EQ(frame.get(chess::PieceType::PAWN, a0achess::kWhite),
            state.pieces(chess::PieceType::PAWN, chess::Color::WHITE));
  EXPECT_EQ(frame.get(chess::PieceType::PAWN, a0achess::kBlack),
            state.pieces(chess::PieceType::PAWN, chess::Color::BLACK));

  EXPECT_EQ(frame.get(chess::PieceType::KNIGHT, a0achess::kWhite),
            state.pieces(chess::PieceType::KNIGHT, chess::Color::WHITE));
  EXPECT_EQ(frame.get(chess::PieceType::KNIGHT, a0achess::kBlack),
            state.pieces(chess::PieceType::KNIGHT, chess::Color::BLACK));

  EXPECT_EQ(frame.get(chess::PieceType::BISHOP, a0achess::kWhite),
            state.pieces(chess::PieceType::BISHOP, chess::Color::WHITE));
  EXPECT_EQ(frame.get(chess::PieceType::BISHOP, a0achess::kBlack),
            state.pieces(chess::PieceType::BISHOP, chess::Color::BLACK));

  EXPECT_EQ(frame.get(chess::PieceType::ROOK, a0achess::kWhite),
            state.pieces(chess::PieceType::ROOK, chess::Color::WHITE));
  EXPECT_EQ(frame.get(chess::PieceType::ROOK, a0achess::kBlack),
            state.pieces(chess::PieceType::ROOK, chess::Color::BLACK));

  EXPECT_EQ(frame.get(chess::PieceType::QUEEN, a0achess::kWhite),
            state.pieces(chess::PieceType::QUEEN, chess::Color::WHITE));
  EXPECT_EQ(frame.get(chess::PieceType::QUEEN, a0achess::kBlack),
            state.pieces(chess::PieceType::QUEEN, chess::Color::BLACK));

  EXPECT_EQ(frame.get(chess::PieceType::KING, a0achess::kWhite),
            state.pieces(chess::PieceType::KING, chess::Color::WHITE));
  EXPECT_EQ(frame.get(chess::PieceType::KING, a0achess::kBlack),
            state.pieces(chess::PieceType::KING, chess::Color::BLACK));
}

TEST(InputFrame, ToStateUnsafeConsistency) {
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

  InputFrame frame(state);
  State unsafe_state = frame.to_state_unsafe();

  EXPECT_EQ(unsafe_state.enpassantSq(), state.enpassantSq());
  EXPECT_EQ(unsafe_state.castlingRights(), state.castlingRights());

  EXPECT_EQ(unsafe_state.pieces(chess::PieceType::PAWN, chess::Color::WHITE),
            state.pieces(chess::PieceType::PAWN, chess::Color::WHITE));
  EXPECT_EQ(unsafe_state.pieces(chess::PieceType::PAWN, chess::Color::BLACK),
            state.pieces(chess::PieceType::PAWN, chess::Color::BLACK));

  EXPECT_EQ(unsafe_state.pieces(chess::PieceType::KNIGHT, chess::Color::WHITE),
            state.pieces(chess::PieceType::KNIGHT, chess::Color::WHITE));
  EXPECT_EQ(unsafe_state.pieces(chess::PieceType::KNIGHT, chess::Color::BLACK),
            state.pieces(chess::PieceType::KNIGHT, chess::Color::BLACK));

  EXPECT_EQ(unsafe_state.pieces(chess::PieceType::BISHOP, chess::Color::WHITE),
            state.pieces(chess::PieceType::BISHOP, chess::Color::WHITE));
  EXPECT_EQ(unsafe_state.pieces(chess::PieceType::BISHOP, chess::Color::BLACK),
            state.pieces(chess::PieceType::BISHOP, chess::Color::BLACK));

  EXPECT_EQ(unsafe_state.pieces(chess::PieceType::ROOK, chess::Color::WHITE),
            state.pieces(chess::PieceType::ROOK, chess::Color::WHITE));
  EXPECT_EQ(unsafe_state.pieces(chess::PieceType::ROOK, chess::Color::BLACK),
            state.pieces(chess::PieceType::ROOK, chess::Color::BLACK));

  EXPECT_EQ(unsafe_state.pieces(chess::PieceType::QUEEN, chess::Color::WHITE),
            state.pieces(chess::PieceType::QUEEN, chess::Color::WHITE));
  EXPECT_EQ(unsafe_state.pieces(chess::PieceType::QUEEN, chess::Color::BLACK),
            state.pieces(chess::PieceType::QUEEN, chess::Color::BLACK));

  EXPECT_EQ(unsafe_state.pieces(chess::PieceType::KING, chess::Color::WHITE),
            state.pieces(chess::PieceType::KING, chess::Color::WHITE));
  EXPECT_EQ(unsafe_state.pieces(chess::PieceType::KING, chess::Color::BLACK),
            state.pieces(chess::PieceType::KING, chess::Color::BLACK));

  for (int i = 0; i < 64; ++i) {
    chess::Square sq = chess::Square(i);
    EXPECT_EQ(unsafe_state.at(sq), state.at(sq));
  }
}

TEST(InputFrame, ToStateUnsafeBlackToMoveNoEpNoCastling) {
  State state;
  const std::string board_str =
    "   a b c d e f g h\n"
    " 8| | | | |k| | | |\n"
    " 7| | | | | | | | |\n"
    " 6| | | | | | | | |\n"
    " 5| | | | | | | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2| | | | | | | | |\n"
    " 1| | | | |K| | |R|\n"
    " b - - 0 1\n";

  std::string fen = convert_to_fen(board_str);
  state.setFen(fen);

  InputFrame frame(state);
  State unsafe_state = frame.to_state_unsafe();

  EXPECT_EQ(unsafe_state.sideToMove(), state.sideToMove());
  EXPECT_EQ(unsafe_state.sideToMove(), chess::Color::BLACK);
  EXPECT_EQ(unsafe_state.enpassantSq(), state.enpassantSq());
  EXPECT_EQ(unsafe_state.castlingRights(), state.castlingRights());

  for (int i = 0; i < 64; ++i) {
    chess::Square sq = chess::Square(i);
    EXPECT_EQ(unsafe_state.at(sq), state.at(sq));
  }
}

TEST(InputFrame, ToStateUnsafePartialCastling) {
  State state;
  const std::string board_str =
    "   a b c d e f g h\n"
    " 8|r| | | |k| | | |\n"
    " 7|p|p|p|p|p|p|p|p|\n"
    " 6| | | | | | | | |\n"
    " 5| | | | | | | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2|P|P|P|P|P|P|P|P|\n"
    " 1| | | | |K| | |R|\n"
    " w Kq - 0 5\n";

  std::string fen = convert_to_fen(board_str);
  state.setFen(fen);

  InputFrame frame(state);
  State unsafe_state = frame.to_state_unsafe();

  EXPECT_EQ(unsafe_state.sideToMove(), state.sideToMove());
  EXPECT_EQ(unsafe_state.enpassantSq(), state.enpassantSq());
  EXPECT_EQ(unsafe_state.castlingRights(), state.castlingRights());

  for (int i = 0; i < 64; ++i) {
    chess::Square sq = chess::Square(i);
    EXPECT_EQ(unsafe_state.at(sq), state.at(sq));
  }
}

TEST(InputFrame, ToStateUnsafeSparseEndgame) {
  State state;
  const std::string board_str =
    "   a b c d e f g h\n"
    " 8| | | | | | | | |\n"
    " 7| | | | | | | | |\n"
    " 6| | | |k| | | | |\n"
    " 5| | | | | | | | |\n"
    " 4| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 2| | | | | | | | |\n"
    " 1| | | | |K| | |R|\n"
    " w - - 0 40\n";

  std::string fen = convert_to_fen(board_str);
  state.setFen(fen);

  InputFrame frame(state);
  State unsafe_state = frame.to_state_unsafe();

  EXPECT_EQ(unsafe_state.sideToMove(), state.sideToMove());
  EXPECT_EQ(unsafe_state.enpassantSq(), state.enpassantSq());
  EXPECT_EQ(unsafe_state.castlingRights(), state.castlingRights());

  for (int i = 0; i < 64; ++i) {
    chess::Square sq = chess::Square(i);
    EXPECT_EQ(unsafe_state.at(sq), state.at(sq));
  }
}

TEST(InputFrame, ToStateUnsafeBlackToMoveWithEp) {
  State state;
  // After 1. e4 d5 2. e5 f5, en passant on f6 with black to move...
  // Actually: white played e5, black played f5, so ep is f6 and white to move.
  // For black-to-move ep: white plays d4, so ep = d3.
  const std::string board_str =
    "   a b c d e f g h\n"
    " 8|r|n|b|q|k|b|n|r|\n"
    " 7|p|p|p| |p|p|p|p|\n"
    " 6| | | | | | | | |\n"
    " 5| | | | | | | | |\n"
    " 4| | | |P|p| | | |\n"
    " 3| | | | | | | | |\n"
    " 2|P|P|P| |P|P|P|P|\n"
    " 1|R|N|B|Q|K|B|N|R|\n"
    " b KQkq d3 0 2\n";

  std::string fen = convert_to_fen(board_str);
  state.setFen(fen);

  InputFrame frame(state);
  State unsafe_state = frame.to_state_unsafe();

  EXPECT_EQ(unsafe_state.sideToMove(), state.sideToMove());
  EXPECT_EQ(unsafe_state.sideToMove(), chess::Color::BLACK);
  EXPECT_EQ(unsafe_state.enpassantSq(), state.enpassantSq());
  EXPECT_EQ(unsafe_state.castlingRights(), state.castlingRights());

  for (int i = 0; i < 64; ++i) {
    chess::Square sq = chess::Square(i);
    EXPECT_EQ(unsafe_state.at(sq), state.at(sq));
  }
}

// ============================================================================
// Syzygy tablebase tests
// ============================================================================

TEST(SyzygyTable, Constants) { EXPECT_EQ(SyzygyTable::kMaxNumPieces, 5); }

TEST(SyzygyTable, TooManyPieces) {
  // Starting position has 32 pieces
  State state;
  state.init();
  auto result = SyzygyTable::instance().fast_lookup(state);
  EXPECT_EQ(result, SyzygyTable::kUnknownResult);
}

TEST(SyzygyTable, CastlingRightsReturnUnknown) {
  // KR vs K but with castling rights still set — should return unknown
  State state("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1");
  auto result = SyzygyTable::instance().fast_lookup(state);
  EXPECT_EQ(result, SyzygyTable::kUnknownResult);
}

TEST(SyzygyTable, KQvK_WhiteWins) {
  State state("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
  auto result = SyzygyTable::instance().fast_lookup(state);
  EXPECT_EQ(result, SyzygyTable::kWhiteWins);
}

TEST(SyzygyTable, KQvK_WhiteWins_IgnoreRule50) {
  // Rule 50 should not affect the result since we're using fast_lookup()
  State state("4k3/8/8/8/8/8/8/3QK3 w - - 99 1");
  auto result = SyzygyTable::instance().fast_lookup(state);
  EXPECT_EQ(result, SyzygyTable::kWhiteWins);
}

TEST(SyzygyTable, KQvK_BlackToMove) {
  // Same material, black to move — still white wins
  State state("4k3/8/8/8/8/8/8/3QK3 b - - 0 1");
  auto result = SyzygyTable::instance().fast_lookup(state);
  EXPECT_EQ(result, SyzygyTable::kWhiteWins);
}

TEST(SyzygyTable, KRvK_WhiteWins) {
  State state("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
  auto result = SyzygyTable::instance().fast_lookup(state);
  EXPECT_EQ(result, SyzygyTable::kWhiteWins);
}

TEST(SyzygyTable, KvKQ_BlackWins) {
  State state("3qk3/8/8/8/8/8/8/4K3 w - - 0 1");
  auto result = SyzygyTable::instance().fast_lookup(state);
  EXPECT_EQ(result, SyzygyTable::kBlackWins);
}

TEST(SyzygyTable, KBvK_Draw) {
  State state("4k3/8/8/8/8/8/8/2B1K3 w - - 0 1");
  auto result = SyzygyTable::instance().fast_lookup(state);
  EXPECT_EQ(result, SyzygyTable::kDraw);
}

TEST(SyzygyTable, KNvK_Draw) {
  State state("4k3/8/8/8/8/8/8/1N2K3 w - - 0 1");
  auto result = SyzygyTable::instance().fast_lookup(state);
  EXPECT_EQ(result, SyzygyTable::kDraw);
}

TEST(SyzygyTable, KvK_Draw) {
  State state("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
  auto result = SyzygyTable::instance().fast_lookup(state);
  EXPECT_EQ(result, SyzygyTable::kDraw);
}

TEST(SyzygyTable, KRvKR_Draw) {
  State state("4k3/8/1r6/8/8/8/6R1/4K3 w - - 0 1");
  auto result = SyzygyTable::instance().fast_lookup(state);
  EXPECT_EQ(result, SyzygyTable::kDraw) << "result=" << static_cast<int>(result);
}

TEST(SyzygyTable, KQvKR_WhiteWins) {
  State state("r3k3/8/8/8/8/8/8/3QK3 w - - 0 1");
  auto result = SyzygyTable::instance().fast_lookup(state);
  EXPECT_EQ(result, SyzygyTable::kWhiteWins);
}

TEST(SyzygyTable, KBBvK_WhiteWins) {
  State state("4k3/8/8/8/8/8/8/2BBK3 w - - 0 1");
  auto result = SyzygyTable::instance().fast_lookup(state);
  EXPECT_EQ(result, SyzygyTable::kWhiteWins);
}

TEST(SyzygyTable, KNNvK_Draw) {
  State state("4k3/8/8/8/8/8/8/1NN1K3 w - - 0 1");
  auto result = SyzygyTable::instance().fast_lookup(state);
  EXPECT_EQ(result, SyzygyTable::kDraw);
}

TEST(SyzygyTable, KQRvKR_WhiteWins) {
  State state("r3k3/8/8/8/8/8/8/R2QK3 w - - 0 1");
  auto result = SyzygyTable::instance().fast_lookup(state);
  EXPECT_EQ(result, SyzygyTable::kWhiteWins);
}

TEST(SyzygyTable, KPPvKP_VariousResults) {
  State state("4k3/4p3/8/8/8/8/3PP3/4K3 w - - 0 1");
  auto result = SyzygyTable::instance().fast_lookup(state);
  EXPECT_NE(result, SyzygyTable::kUnknownResult);
}

TEST(SyzygyTable, KQvK_BestAction) {
  State state("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
  Move move;
  auto result = SyzygyTable::instance().slow_lookup(state, &move);
  EXPECT_EQ(result, SyzygyTable::kWhiteWins);

  auto analysis = Game::Rules::analyze(state);
  ASSERT_FALSE(analysis.is_terminal());
  EXPECT_TRUE(analysis.valid_moves().contains(move));
}

TEST(SyzygyTable, KRvK_BestAction) {
  State state("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
  Move move;
  auto result = SyzygyTable::instance().slow_lookup(state, &move);
  EXPECT_EQ(result, SyzygyTable::kWhiteWins);

  auto analysis = Game::Rules::analyze(state);
  ASSERT_FALSE(analysis.is_terminal());
  EXPECT_TRUE(analysis.valid_moves().contains(move));
}

TEST(SyzygyTable, DrawPosition_BestAction) {
  // KR vs KR — drawn, not terminal (unlike KN vs K which is insufficient material)
  State state("3rk3/8/8/8/8/8/8/3RK3 w - - 0 1");
  Move move;
  auto result = SyzygyTable::instance().slow_lookup(state, &move);
  EXPECT_EQ(result, SyzygyTable::kDraw);

  auto analysis = Game::Rules::analyze(state);
  ASSERT_FALSE(analysis.is_terminal());
  EXPECT_TRUE(analysis.valid_moves().contains(move));
}

TEST(SyzygyTable, PawnPromotion_WhiteWins) {
  State state("8/4P3/8/8/8/8/2k5/4K3 w - - 0 1");
  Move move;
  auto result = SyzygyTable::instance().slow_lookup(state, &move);
  EXPECT_EQ(result, SyzygyTable::kWhiteWins);

  auto analysis = Game::Rules::analyze(state);
  ASSERT_FALSE(analysis.is_terminal());
  EXPECT_TRUE(analysis.valid_moves().contains(move));
}

TEST(SyzygyTable, EnPassant) {
  // KP vs KP with en passant available
  State state("4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 1");
  Move move;
  auto result = SyzygyTable::instance().slow_lookup(state, &move);
  EXPECT_NE(result, SyzygyTable::kUnknownResult);

  auto analysis = Game::Rules::analyze(state);
  ASSERT_FALSE(analysis.is_terminal());
  EXPECT_TRUE(analysis.valid_moves().contains(move));
}

TEST(SyzygyTable, HighRule50_RootProbe_IsDraw) {
  // KQ vs K — theoretically winning for white, but halfmove clock at 96 means
  // only 4 half-moves (2 full moves) remain before the 50-move draw. Mating with
  // KQ vs K requires ~10 moves, so tb_probe_root should report a draw.
  State state("4k3/8/8/8/8/8/8/3QK3 w - - 96 1");
  Move move;
  auto result = SyzygyTable::instance().slow_lookup(state, &move);
  EXPECT_EQ(result, SyzygyTable::kDraw) << "result=" << static_cast<int>(result);
}

TEST(SyzygyTable, LowRule50_RootProbe_StillWins) {
  // Same position but halfmove clock at 0 — white wins.
  State state("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
  Move move;
  auto result = SyzygyTable::instance().slow_lookup(state, &move);
  EXPECT_EQ(result, SyzygyTable::kWhiteWins);
}

TEST(SyzygyTable, HighRule50_FastLookup_IgnoresRule50) {
  // fast_lookup ignores the 50-move rule — it always passes rule50=0 to tb_probe_wdl.
  // So even with halfmove clock at 96, it reports the theoretical result (white wins).
  State state("4k3/8/8/8/8/8/8/3QK3 w - - 96 1");
  auto result = SyzygyTable::instance().fast_lookup(state);
  EXPECT_EQ(result, SyzygyTable::kWhiteWins) << "result=" << static_cast<int>(result);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
