#include "games/chess/SyzygyTable.hpp"

#include "util/Exceptions.hpp"

#include <filesystem>
#include <mutex>
#include <tbprobe.h>

namespace a0achess {

namespace {

constexpr const char* kSyzygyPath = "/workspace/syzygy";

}  // namespace

inline SyzygyTable& SyzygyTable::instance() {
  static SyzygyTable table;
  return table;
}

inline SyzygyTable::Result SyzygyTable::fast_lookup(const GameState& state) const {
  int num_pieces = state.occ().count();
  if (num_pieces > kMaxNumPieces) return kUnknownResult;

  if (state.castlingRights().has(chess::Color::WHITE) ||
      state.castlingRights().has(chess::Color::BLACK)) {
    return kUnknownResult;
  }

  uint64_t white = state.us(chess::Color::WHITE).getBits();
  uint64_t black = state.us(chess::Color::BLACK).getBits();
  uint64_t kings = state.pieces(chess::PieceType::KING).getBits();
  uint64_t queens = state.pieces(chess::PieceType::QUEEN).getBits();
  uint64_t rooks = state.pieces(chess::PieceType::ROOK).getBits();
  uint64_t bishops = state.pieces(chess::PieceType::BISHOP).getBits();
  uint64_t knights = state.pieces(chess::PieceType::KNIGHT).getBits();
  uint64_t pawns = state.pieces(chess::PieceType::PAWN).getBits();
  unsigned ep = state.enpassantSq() == chess::Square::NO_SQ ? 0 : state.enpassantSq().index();
  bool turn = state.sideToMove() == chess::Color::WHITE;

  unsigned wdl = tb_probe_wdl(white, black, kings, queens, rooks, bishops, knights, pawns,
                              0 /*rule50*/, 0 /*castling*/, ep, turn);
  if (wdl == TB_RESULT_FAILED) return kUnknownResult;

  switch (wdl) {
    case TB_WIN:
      return turn ? kWhiteWins : kBlackWins;
    case TB_LOSS:
      return turn ? kBlackWins : kWhiteWins;
    default:
      return kDraw;
  }
}

inline SyzygyTable::Result SyzygyTable::slow_lookup(const GameState& state, Move* move) const {
  int num_pieces = state.occ().count();
  if (num_pieces > kMaxNumPieces) return kUnknownResult;

  if (state.castlingRights().has(chess::Color::WHITE) ||
      state.castlingRights().has(chess::Color::BLACK)) {
    return kUnknownResult;
  }

  uint64_t white = state.us(chess::Color::WHITE).getBits();
  uint64_t black = state.us(chess::Color::BLACK).getBits();
  uint64_t kings = state.pieces(chess::PieceType::KING).getBits();
  uint64_t queens = state.pieces(chess::PieceType::QUEEN).getBits();
  uint64_t rooks = state.pieces(chess::PieceType::ROOK).getBits();
  uint64_t bishops = state.pieces(chess::PieceType::BISHOP).getBits();
  uint64_t knights = state.pieces(chess::PieceType::KNIGHT).getBits();
  uint64_t pawns = state.pieces(chess::PieceType::PAWN).getBits();
  unsigned ep = state.enpassantSq() == chess::Square::NO_SQ ? 0 : state.enpassantSq().index();
  unsigned rule50 = state.halfMoveClock();
  bool turn = state.sideToMove() == chess::Color::WHITE;

  mit::unique_lock lock(mutex_);
  unsigned result = tb_probe_root(white, black, kings, queens, rooks, bishops, knights, pawns,
                                  rule50, 0 /*castling*/, ep, turn, nullptr);
  lock.unlock();

  if (result == TB_RESULT_FAILED) return kUnknownResult;

  unsigned wdl = TB_GET_WDL(result);
  *move = fathom_result_to_move(state, result);

  switch (wdl) {
    case TB_WIN:
      return turn ? kWhiteWins : kBlackWins;
    case TB_LOSS:
      return turn ? kBlackWins : kWhiteWins;
    default:
      return kDraw;
  }
}

inline SyzygyTable::SyzygyTable() {
  if (!std::filesystem::is_directory(kSyzygyPath)) {
    throw util::CleanException(
      "Syzygy tablebases not found at %s. "
      "Please run setup_wizard.py from outside the Docker container to download them.",
      kSyzygyPath);
  }
  if (!tb_init(kSyzygyPath)) {
    throw util::CleanException("Failed to initialize Syzygy tablebases at %s.", kSyzygyPath);
  }
  if (TB_LARGEST == 0) {
    throw util::CleanException(
      "Syzygy tablebases at %s contain no valid table files. "
      "Please run setup_wizard.py from outside the Docker container to download them.",
      kSyzygyPath);
  }
}

}  // namespace a0achess
