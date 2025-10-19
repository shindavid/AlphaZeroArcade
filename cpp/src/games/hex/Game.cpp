#include "games/hex/Game.hpp"

#include <boost/json/object.hpp>

namespace hex {

void Game::Rules::apply(State& state, core::action_t action) {
  static constexpr auto B = Constants::kBoardDim;
  auto cp = state.core.cur_player;

  if (action == kSwap) {
    DEBUG_ASSERT(cp == Constants::kBlue && !state.core.post_swap_phase,
                 "Swap action can only be applied by Blue before the swap phase");

    core::action_t prev_action = state.core.find_occupied(Constants::kRed);
    Rules::init_state(state);
    state.core.cur_player = Constants::kSecondPlayer;
    apply(state, compute_mirror_action(prev_action));
  } else {
    int8_t row = action / B;
    int8_t col = action % B;
    State::Core& core = state.core;
    State::Aux& aux = state.aux;

    DEBUG_ASSERT(!(core.rows[cp][row] & (mask_t(1) << col)),
                 "Cannot place a piece on an already occupied cell");
    core.rows[cp][row] |= (mask_t(1) << col);

    constexpr int kMaxNumNeighbors = 6;
    struct neighbor_t {
      void init(int8_t r, int8_t c) {
        row = r;
        col = c;
      }
      int8_t row;
      int8_t col;
    };
    neighbor_t neighbors[kMaxNumNeighbors];
    int num_neighbors = 0;

    auto init = [&](int8_t r, int8_t c, bool use) {
      neighbors[num_neighbors].init(r, c);
      num_neighbors += use;
    };

    init(row + 1, col, row + 1 < B);
    init(row - 1, col, row > 0);
    init(row, col + 1, col + 1 < B);
    init(row, col - 1, col > 0);
    init(row - 1, col + 1, row > 0 && col + 1 < B);
    init(row + 1, col - 1, row + 1 < B && col > 0);

    // filter out neighbors that are not occupied by the current player
    int w = 0;
    for (int n = 0; n < num_neighbors; ++n) {
      int nr = neighbors[n].row;
      int nc = neighbors[n].col;
      bool occupied = core.rows[cp][nr] & (mask_t(1) << nc);
      neighbors[w] = neighbors[n];
      w += occupied;
    }
    int num_neighbors_to_unite = w;

    auto& U = aux.union_find[cp];

    for (int n = 0; n < num_neighbors_to_unite; ++n) {
      int nr = neighbors[n].row;
      int nc = neighbors[n].col;
      U.unite(action, to_vertex(nr, nc));
    }

    if (cp == Constants::kRed) {  // red connects N to S
      if (row == 0) {
        U.unite(action, hex::UnionFind::kVirtualVertex1);
      } else if (row == B - 1) {
        U.unite(action, hex::UnionFind::kVirtualVertex2);
      }
    } else {  // blue connects W to E
      if (col == 0) {
        U.unite(action, hex::UnionFind::kVirtualVertex1);
      } else if (col == B - 1) {
        U.unite(action, hex::UnionFind::kVirtualVertex2);
      }
    }
  }

  state.core.post_swap_phase |= (cp == Constants::kBlue);
  state.core.cur_player = 1 - cp;
}

void Game::IO::print_state(std::ostream& ss, const State& state, core::action_t last_action,
                           const Types::player_name_array_t* player_names) {
  /*
               A B C D E F G H I J K
          11 / / / / / / / / / / / / 11
         10 / / / / / / / / / / / / 10
         9 / / / / / / / / / / / /  9
        8 / / / / / / / / / / / /  8
       7 / / / / / / / / / / / /  7
      6 / / / / /W/ / / / / / /  6
     5 / / / / /B/ / / / / / /  5
    4 / / / / / / / / / / / /  4
   3 / / / /B/ / / / / / / /  3
  2 / / / / / / / / / / / /  2
 1 / / / / / / / / / / / /  1
   A B C D E F G H I J K

*/
  constexpr int B = Constants::kBoardDim;

  bool display_last_action = last_action >= 0 && last_action != kSwap;
  int blink_row = -1;
  int blink_col = -1;
  if (display_last_action) {
    blink_row = last_action / B;
    blink_col = last_action % B;
  }

  constexpr int buf_size = 4096;
  char buffer[buf_size];
  int cx = 0;

  cx += snprintf(buffer + cx, buf_size - cx, "               %sA B C D E F G H I J K%s\n",
                 ansi::kRed(""), ansi::kReset(""));
  for (int row = B - 1; row >= 0; --row) {
    cx += print_row(buffer + cx, buf_size - cx, state, row, row == blink_row ? blink_col : -1);
  }
  cx += snprintf(buffer + cx, buf_size - cx, "   %sA B C D E F G H I J K%s\n", ansi::kRed(""),
                 ansi::kReset(""));

  if (player_names) {
    cx += snprintf(buffer + cx, buf_size - cx, "\n");
    for (int p = 0; p < Constants::kNumPlayers; ++p) {
      cx += snprintf(buffer + cx, buf_size - cx, "%s: %s\n", player_to_str(p).c_str(),
                     (*player_names)[p].c_str());
    }
  }

  RELEASE_ASSERT(cx < buf_size, "Buffer overflow ({} < {})", cx, buf_size);
  ss << buffer << std::endl;
}

boost::json::array Game::IO::state_to_json(const State& state) {
  boost::json::array arr;
  for (int row = 0; row < Constants::kBoardDim; ++row) {
    for (int col = 0; col < Constants::kBoardDim; ++col) {
      arr.push_back(state.get_player_at(row, col));
    }
  }
  return arr;
}

int Game::IO::print_row(char* buf, int n, const State& state, int row, int blink_column) {
  int cx = 0;

  // print (row) blank spaces
  for (int i = 0; i < row; ++i) {
    buf[cx] = ' ';
    cx += cx < n;
  }
  cx += snprintf(buf + cx, n - cx, "%s%2d%s /", ansi::kBlue(""), row + 1, ansi::kReset(""));

  mask_t row_masks[Constants::kNumPlayers];
  row_masks[Constants::kRed] = state.core.rows[Constants::kRed][row];
  row_masks[Constants::kBlue] = state.core.rows[Constants::kBlue][row];

  for (int col = 0; col < Constants::kBoardDim; ++col) {
    const char* a = "";
    const char* b = "";
    const char* c = " ";
    const char* d = "";
    if (row_masks[Constants::kRed] & (mask_t(1) << col)) {
      b = ansi::kRed("");
      c = ansi::kCircle("R");
      d = ansi::kReset("");
    } else if (row_masks[Constants::kBlue] & (mask_t(1) << col)) {
      b = ansi::kBlue("");
      c = ansi::kCircle("B");
      d = ansi::kReset("");
    }
    if (col == blink_column) {
      a = ansi::kBlink("");
    }

    cx += snprintf(buf + cx, n - cx, "%s%s%s%s/", a, b, c, d);
  }

  cx += snprintf(buf + cx, n - cx, "%s%3d%s\n", ansi::kBlue(""), row + 1, ansi::kReset(""));
  return cx;
}

}  // namespace hex
