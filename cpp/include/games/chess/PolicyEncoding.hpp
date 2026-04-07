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

  static constexpr int promo_bases[9] = {0,  6,  15, 24, 33,
                                         42, 51, 60, 66};  // indexed by from_square % 8

  int encode(const Move& move, chess::Color side_to_move) const;
  Move decode(int index, const chess::Board& board) const;

 private:
  struct Data {
    uint64_t bitmap;
    int offset;

    bool operator<(const Data& other) const { return offset < other.offset; }
  };

  static int count_before_k(uint64_t bitmap, int k) {
    return std::popcount(bitmap & ((1ULL << k) - 1));
  }
  MoveData decode_move_data(int index) const;

  Data data_[64];  // Indexed by from_square
  int num_non_promo_moves_;
};

struct PolicyEncoding {
  using Game = a0achess::Game;
  using State = Game::State;
  using Shape = Eigen::Sizes<Game::Constants::kNumMoves>;
  using Tensor = eigen_util::FTensor<Shape>;
  static constexpr int kRank = eigen_util::extract_rank_v<Shape>;
  using Index = Eigen::array<Eigen::Index, kRank>;

  static Index to_index(const InputFrame& frame, const Move& move);
  static Move to_move(const State& state, const Index& index);
  static inline MoveEncodingTable move_encoding_table;
};

}  // namespace a0achess

#include "inline/games/chess/PolicyEncoding.inl"
