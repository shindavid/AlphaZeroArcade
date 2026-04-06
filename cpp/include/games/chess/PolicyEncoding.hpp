#pragma once

#include "games/chess/Game.hpp"
#include "games/chess/InputFrame.hpp"
#include "games/chess/Move.hpp"

namespace a0achess {

class MoveEncodingTable {
 public:
  using State = Game::State;

  struct MoveData {
    int from_square;
    int to_square;
    chess::PieceType pt = chess::PieceType::NONE;
  };

  MoveEncodingTable();

  static constexpr int promo_bases[9] = {0, 6, 15, 24, 33, 42, 51, 60, 66};  // indexed by from_square % 8

  int encode(const Move& move, chess::Color side_to_move) const;
  Move decode(int index, const chess::Board& board) const;

 private:
  using move_table_t = std::array<MoveData, 1858>;
  struct Data {
    uint64_t bitmap;
    int offset;
  };
  static int count_before_k(uint64_t bitmap, int k) {
    return std::popcount(bitmap & ((1ULL << k) - 1));
  }
  Data data_[64];  // Indexed by from_square
  move_table_t move_table_;
};

struct PolicyEncoding {
  using Game = a0achess::Game;
  using Shape = Eigen::Sizes<Game::Constants::kNumMoves>;
  using Tensor = eigen_util::FTensor<Shape>;
  static constexpr int kRank = eigen_util::extract_rank_v<Shape>;
  using Index = Eigen::array<Eigen::Index, kRank>;

  static Index to_index(const InputFrame& frame, const Move& move);
  static Move to_move(const InputFrame& frame, const Index& index);
  static inline MoveEncodingTable move_encoding_table;
};

}  // namespace a0achess

#include "inline/games/chess/PolicyEncoding.inl"
