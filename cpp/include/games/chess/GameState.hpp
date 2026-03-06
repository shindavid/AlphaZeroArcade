#pragma once

#include "chess-library/include/chess.hpp"
#include "core/BasicTypes.hpp"
#include "games/chess/MoveEncoder.hpp"
#include "util/StaticCircularBuffer.hpp"

namespace a0achess {

class GameState {
 public:
  using zobrist_hash_t = uint64_t;
  using PastZobristHashes = util::StaticCircularBuffer<zobrist_hash_t, 8>;

  GameState() = default;
  GameState(const chess::Board& board) : board_(board) {}

  auto operator<=>(const GameState& other) const { return board_.hash() <=> other.board_.hash(); }
  auto operator==(const GameState& other) const { return board_.hash() == other.board_.hash(); }
  zobrist_hash_t hash() const { return board_.hash(); }

  void reset();
  chess::Movelist generate_legal_moves() const;
  core::action_t move_to_action(const chess::Move& move) const { return move_to_nn_idx(board_, move); }
  chess::Move action_to_move(core::action_t action) const { return nn_idx_to_move(board_, action); }
  chess::Color side_to_move() const { return board_.sideToMove(); }
  void apply_action(core::action_t action);
  bool in_check() const { return board_.inCheck(); }
  bool is_insufficient_material() const { return board_.isInsufficientMaterial(); }
  bool is_half_move_draw() const { return board_.isHalfMoveDraw(); }
  bool is_repetition(int repetitions) const;
  int half_move_clock() const { return board_.halfMoveClock(); }
  chess::Board::CastlingRights castling_rights() const { return board_.castlingRights(); }
  uint64_t pieces_bb(chess::PieceType pt, chess::Color c) const { return board_.pieces(pt, c).getBits(); }
  std::string fen() const { return board_.getFen(); }
  core::action_t action_from_uci(const std::string& uci) const;

 private:
  chess::Board board_;
  PastZobristHashes past_hashes_;
};

}  // namespace a0achess

#include "inline/games/chess/GameState.inl"
