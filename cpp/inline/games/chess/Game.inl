#include "games/chess/Game.hpp"
#include "core/BasicTypes.hpp"
#include "core/DefaultCanonicalizer.hpp"
#include "lc0/chess/board.h"
#include "lc0/neural/encoder.h"


namespace chess {

inline void Game::Rules::init_state(State& state) {
  state.board = lczero::ChessBoard::kStartposBoard;
  state.recent_hashes.clear();
  state.zobrist_hash = state.board.Hash();
  state.history_hash = state.zobrist_hash;
  state.rule50_ply = 0;
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const State& state) {
  const auto legal_moves = state.board.GenerateLegalMoves();
  Game::Types::ActionMask mask;
  for (const auto& move : legal_moves) {
    core::action_t action = static_cast<core::action_t>(lczero::MoveToNNIndex(move, 0));
    mask.set(action);
  }
  return mask;
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.board.flipped() ? kBlack : kWhite;
}

inline void Game::Rules::apply(State& state, core::action_t action) {
  auto move = lczero::MoveFromNNIndex(action, 0);
  bool reset_50_moves = state.board.ApplyMove(move);
  state.board.Mirror();

  // TODO: Optimization: only store the last board hash of the same player
  // (since only those are relevant for threefold repetition)
  // store the opponent's board hash for the next state
  state.recent_hashes.push_back(state.zobrist_hash);

  // TODO: Implement Zobrist hashing (lc0's hash is not Zobrist)
  state.zobrist_hash = state.board.Hash();

  if (reset_50_moves) {
    state.rule50_ply = 0;
    state.history_hash = state.zobrist_hash;
    state.recent_hashes.clear();
  } else {
    state.rule50_ply++;
    state.history_hash = lczero::HashCat({state.history_hash, state.zobrist_hash});
  }
}

inline bool Game::Rules::is_terminal(const State& state, core::seat_index_t, core::action_t,
                                     GameResults::Tensor& outcome) {
  const auto& board = state.board;
  auto legal_moves = board.GenerateLegalMoves();
  if (legal_moves.empty()) {
    if (board.IsUnderCheck()) {
      core::seat_index_t cp = get_current_player(state);
      outcome = core::WinLossDrawResults::win(1 - cp);
      return true;
    }
    // Stalemate.
    outcome = core::WinLossDrawResults::draw();
    return true;
  }

  if (!board.HasMatingMaterial()) {
    outcome = core::WinLossDrawResults::draw();
    return true;
  }

  if (state.rule50_ply >= 100) {
    outcome = core::WinLossDrawResults::draw();
    return true;
  }

  if (state.count_repetitions() >= 2) {
    outcome = core::WinLossDrawResults::draw();
    return true;
  }

  return false;
}

inline std::string Game::IO::action_to_str(core::action_t action, core::action_mode_t) {
  return lczero::MoveFromNNIndex(action, 0).ToString(false);
}

inline Game::Types::SymmetryMask Game::Symmetries::get_mask(const State& state) {
  Types::SymmetryMask mask;
  mask.set(0);

  if (state.board.pawns().empty() && state.board.castlings().no_legal_castle()) {
    mask.set();
  }
  return mask;
}

inline void Game::Symmetries::apply(State& state, group::element_t sym) {
  using namespace bitmap_util;
  using D4 = groups::D4;
  auto& s = state;
  switch (sym) {
    case D4::kIdentity:
      return;
    case D4::kRot90:
      return s.board.rot90_clockwise();
    case D4::kRot180:
      return s.board.rot180();
    case D4::kRot270:
      return s.board.rot270_clockwise();
    case D4::kFlipVertical:
      return s.board.flip_vertical();
    case D4::kFlipMainDiag:
      return s.board.flip_main_diag();
    case D4::kMirrorHorizontal:
      return s.board.mirror_horizontal();
    case D4::kFlipAntiDiag:
      return s.board.flip_anti_diag();
    default:
      throw util::Exception("Unknown group element: {}", sym);
  }
}

template <eigen_util::concepts::FTensor Tensor>
inline void Game::Symmetries::apply(Tensor& tensor, group::element_t sym, core::action_mode_t) {
  using namespace eigen_util;
  using D4 = groups::D4;
  constexpr int N = kBoardDim;
  switch (sym) {
    case D4::kIdentity:
      return;
    case D4::kRot90:
      return rot90_clockwise<N>(tensor);
    case D4::kRot180:
      return rot180<N>(tensor);
    case D4::kRot270:
      return rot270_clockwise<N>(tensor);
    case D4::kFlipVertical:
      return flip_vertical<N>(tensor);
    case D4::kFlipMainDiag:
      return flip_main_diag<N>(tensor);
    case D4::kMirrorHorizontal:
      return mirror_horizontal<N>(tensor);
    case D4::kFlipAntiDiag:
      return flip_anti_diag<N>(tensor);
    default:
      throw util::Exception("Unknown group element: {}", sym);
  }
}

inline group::element_t Game::Symmetries::get_canonical_symmetry(const State& state) {
  using DefaultCanonicalizer = core::DefaultCanonicalizer<Game>;
  return DefaultCanonicalizer::get(state);
}

}  // namespace chess
