#include "games/chess/Game.hpp"

#include <chess-library/include/chess.hpp>

#include <array>
#include <string>

namespace a0achess {

void Game::IO::print_state(std::ostream& ss, const State& state, core::action_t last_action,
                           const Types::player_name_array_t* player_names) {
  std::array<char, 64> buf;
  buf.fill(' ');

  InputFrame frame(state);

  for (chess::PieceType piece_type :
       {chess::PieceType::PAWN, chess::PieceType::KNIGHT, chess::PieceType::BISHOP,
        chess::PieceType::ROOK, chess::PieceType::QUEEN, chess::PieceType::KING}) {
    for (core::seat_index_t player = 0; player < Constants::kNumPlayers; ++player) {
      chess::Color color = (player == kWhite) ? chess::Color::WHITE : chess::Color::BLACK;
      chess::Bitboard bb = frame.get(piece_type, player);
      chess::Piece piece(piece_type, color);
      while (bb) {
        chess::Square sq = bb.pop();
        buf[sq.index()] = ((std::string)piece).front();
      }
    }
  }

  // print the board, include coordinates
  ss << "   a b c d e f g h\n";
  for (int rank = 7; rank >= 0; --rank) {
    ss << " " << (rank + 1) << "|";
    for (int file = 0; file < 8; ++file) {
      ss << buf[rank * 8 + file] << "|";
    }
    ss << (rank + 1) << "\n";
  }
  ss << "   a b c d e f g h\n\n";

  if (player_names) {
    for (core::seat_index_t player = 0; player < Constants::kNumPlayers; ++player) {
      ss << "Player " << kSeatChars[player] << ": " << (*player_names)[player];
      chess::Color color = (player == kWhite) ? chess::Color::WHITE : chess::Color::BLACK;
      if (last_action >= 0 && color != state.sideToMove()) {
        ss << " " << action_to_str(last_action);
      }
      ss << "\n";
    }
    ss << "\n";
  }
}

}  // namespace a0achess
