#pragma once

#include <games/blokus/Constants.hpp>

namespace blokus {

class Piece {
public:
  Piece(piece_id_t id, const std::vector<std::vector<bool>>& shape)
    : id_(id), shape_(shape) {}

private:
  piece_id_t id_;
  std::vector<std::vector<bool>> shape_;
};

}  // namespace blokus
