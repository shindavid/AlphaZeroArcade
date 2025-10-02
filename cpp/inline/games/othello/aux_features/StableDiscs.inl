#include "games/othello/aux_features/StableDiscs.hpp"

#include <iostream>

namespace othello {

inline mask_t compute_stable_discs(mask_t cur_player_mask, mask_t opponent_mask,
                                   const mask_t& stable = 0) {
  mask_t stable_curr = cur_player_mask & stable;
  mask_t stable_oppo = opponent_mask & stable;
  mask_t stable_edge = get_stable_edge(cur_player_mask, opponent_mask);
  stable_curr |= stable_edge & cur_player_mask;
  stable_oppo |= stable_edge & opponent_mask;

  full_axes_stable_discs(cur_player_mask, opponent_mask, stable_curr, stable_oppo);

  stable_curr |= extend_stable_frontier(cur_player_mask, stable_curr);
  stable_oppo |= extend_stable_frontier(opponent_mask, stable_oppo);

  return stable_curr | stable_oppo;
}

inline mask_t extend_stable_frontier(mask_t mask, mask_t stable) {
  auto ray_grow = [&](auto shift) {
    mask_t out = 0;
    mask_t frontier = stable;
    while (frontier) {
      frontier = shift(frontier) & mask & ~out;
      out |= frontier;
    }
    return out;
  };

  while (true) {
    const mask_t e = ray_grow(step_east);
    const mask_t w = ray_grow(step_west);
    const mask_t n = ray_grow(step_north);
    const mask_t s = ray_grow(step_south);
    const mask_t ne = ray_grow(step_northeast);
    const mask_t sw = ray_grow(step_southwest);
    const mask_t nw = ray_grow(step_northwest);
    const mask_t se = ray_grow(step_southeast);

    mask_t horiz, vert, diag1, diag2;

    horiz = e | w;
    vert = n | s;
    diag1 = ne | sw;
    diag2 = nw | se;

    const mask_t candidates = horiz & vert & diag1 & diag2;
    const mask_t newS = candidates & ~stable;
    if (!newS) break;
    stable |= newS;
  }
  return stable;
}

template <size_t N>
inline mask_t union_of_full_lines(mask_t mask, const std::array<mask_t, N>& lines) {
  mask_t out = 0;
  for (size_t i = 0; i < N; ++i) {
    mask_t line = lines[i];
    mask_t fullMask = -((mask & line) == line);
    out |= (line & fullMask);
  }
  return out;
}

inline void full_axes_stable_discs(mask_t cur_player_mask, mask_t opponent_mask,
                                   mask_t& stable_curr, mask_t& stable_oppo) {
  mask_t mask = cur_player_mask | opponent_mask;
  const mask_t ranksFull = union_of_full_lines(mask, kRanks);
  const mask_t filesFull = union_of_full_lines(mask, kFiles);
  const mask_t seFull = union_of_full_lines(mask, kDiagSE);
  const mask_t swFull = union_of_full_lines(mask, kDiagSW);

  const mask_t all_axes_full = ranksFull & filesFull & seFull & swFull;
  stable_curr |= all_axes_full & cur_player_mask;
  stable_oppo |= all_axes_full & opponent_mask;
}

inline line_mask_t pack_file(mask_t bb, int file) {
  // Pack a file (0 = A … 7 = H) into an 8-bit mask: rank 1→bit0, …, rank 8→bit7
  mask_t m = 0;
  for (int r = 0; r < 8; ++r) {  // r = rank index (0..7) == (row)
    int sq = r * 8 + file;       // square index on that file
    if (bb & detail::bit64(sq)) m |= (1u << r);
  }
    return m;
}

inline mask_t unpack_file_middle(line_mask_t m, int file) {
  // Unpack an 8-bit mask back onto a file, BUT only rows 2..7 (r=1..6) to avoid corners.
  // (Corners came from the rank lookups already.)
  mask_t out = 0;
  for (int r = 1; r <= 6; ++r) {  // exclude r=0 (rank1) and r=7 (rank8)
    if (m & (1u << r)) out |= detail::bit64(r * 8 + file);
  }
  return out;
}

inline mask_t get_stable_edge(mask_t P, mask_t O) {
  mask_t stable = 0;
  // Rank 1: bits 0..7
  {
    line_mask_t p8 = static_cast<line_mask_t>(P & 0xFF);
    line_mask_t o8 = static_cast<line_mask_t>(O & 0xFF);
    stable |= static_cast<mask_t>(edge_stable_lookup(p8, o8));
  }

  // Rank 8: bits 56..63
  {
    line_mask_t p8 = static_cast<line_mask_t>((P >> 56) & 0xFF);
    line_mask_t o8 = static_cast<line_mask_t>((O >> 56) & 0xFF);
    stable |= static_cast<mask_t>(edge_stable_lookup(p8, o8)) << 56;
  }

  // File A (file = 0), excluding corners (A1/A8)
  {
    line_mask_t p8 = pack_file(P, 0);
    line_mask_t o8 = pack_file(O, 0);
    line_mask_t m = edge_stable_lookup(p8, o8);
    stable |= unpack_file_middle(m, 0);
  }

  // File H (file = 7), excluding corners (H1/H8)
  {
    line_mask_t p8 = pack_file(P, 7);
    line_mask_t o8 = pack_file(O, 7);
    line_mask_t m = edge_stable_lookup(p8, o8);
    stable |= unpack_file_middle(m, 7);
  }

  return stable;
}

inline void fill_ternary_digits(line_mask_t curr_player_mask, line_mask_t opponent_mask,
                                int digits[8], bool curr_player_is_one) {
  int curr_player_value, opponent_value;
  if (curr_player_is_one) {
    curr_player_value = 1;
    opponent_value = 2;
  } else {
    curr_player_value = 2;
    opponent_value = 1;
  }

  for (int i = 0; i < 8; ++i) {
    if (curr_player_mask & detail::bit8(i)) {
      digits[i] = curr_player_value;
    } else if (opponent_mask & detail::bit8(i)) {
      digits[i] = opponent_value;
    } else {
      digits[i] = 0;
    }
  }
}

inline int ternary_int_value(const int digits[8]) {
  int value = 0;
  for (int i = 0; i < 8; ++i) {
    value += digits[i] * std::pow(3, i);
  }
  return value;
}

inline int to_ternary_value(line_mask_t curr_player_mask, line_mask_t opponent_mask) {
  int digits[8];
  if (curr_player_mask & detail::bit8(7)) {
    fill_ternary_digits(curr_player_mask, opponent_mask, digits, true);
  } else {
    fill_ternary_digits(curr_player_mask, opponent_mask, digits, false);
  }
  return ternary_int_value(digits);
}

inline line_mask_t edge_stable_lookup(line_mask_t curr_player_mask, line_mask_t opponent_mask) {
  return EDGE_STABILITY[to_ternary_value(curr_player_mask, opponent_mask)];
}

}  // namespace othello
