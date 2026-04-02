#include "games/chess/Game.hpp"

#include <chess-library/include/chess.hpp>

#include <array>
#include <string>

namespace a0achess {

void Game::IO::print_state(std::ostream& ss, const State& state, const Move* last_move,
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
      if (last_move && color != state.sideToMove()) {
        ss << " " << move_to_str(*last_move);
      }
      ss << "\n";
    }
    ss << "\n";
  }
}

Move Game::IO::move_from_str(const GameState& state, std::string_view s) {
  auto phase = Game::Rules::get_game_phase(state);
  return Move(chess::uci::uciToMove(state, s).move(), phase);
}

Move Game::IO::deserialize_move(std::string_view s) {
  size_t dot_pos = s.find('.');
  if (dot_pos == std::string_view::npos) {
    throw util::Exception("invalid move string: {}", s);
  }
  core::game_phase_t phase = util::atoi(s.substr(0, dot_pos));
  int move_int = util::atoi(s.substr(dot_pos + 1));
  return Move(chess::Move(move_int), phase);
}

}  // namespace a0achess
