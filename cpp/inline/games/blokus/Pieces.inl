#include <games/blokus/Pieces.hpp>

#include <util/Asserts.hpp>

#include <cstring>

namespace blokus {

inline PieceOrientation::PieceOrientation(piece_orientation_index_t poi, piece_index_t pi,
                                          group::element_t e, column_mask_t c0, column_mask_t c1,
                                          column_mask_t c2, column_mask_t c3, column_mask_t c4) {
  index = poi;
  piece_index = pi;
  orientation = e;
  column_masks[0] = c0;
  column_masks[1] = c1;
  column_masks[2] = c2;
  column_masks[3] = c3;
  column_masks[4] = c4;
}

inline Piece::Piece(const char* n, piece_index_t pi, piece_orientation_index_t canonical_poi,
                    group_subset_t s, column_mask_t c0, column_mask_t c1, column_mask_t c2) {
  util::release_assert(strlen(n) == 2);
  std::memcpy(name, n, 3);
  index = pi;
  canonical_orientation_index = canonical_poi;
  orientations = s;
  column_masks[0] = c0;
  column_masks[1] = c1;
  column_masks[2] = c2;
}

inline oriented_piece_index_t Piece::orient(group::element_t sym) const {
  util::debug_assert(orientations & (1 << sym));
  return canonical_orientation_index + std::popcount(orientations & ((1 << sym) - 1));
}

}  // namespace blokus
