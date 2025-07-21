#include "games/othello/EdaxOracle.hpp"

#include "util/StringUtil.hpp"

#include <format>

namespace othello {

EdaxOracle::EdaxOracle(bool verbose, bool deterministic_mode)
    : verbose_(verbose), deterministic_mode_(deterministic_mode) {
  line_buffer_.resize(1);

  auto edax_dir = boost::filesystem::path("extra_deps/edax-reversi");
  auto edax_relative_bin = boost::filesystem::path("bin/lEdax-x64-modern");
  auto edax_bin = edax_dir / edax_relative_bin;

  if (!boost::filesystem::is_regular_file(edax_bin)) {
    throw util::CleanException("File does not exist: {}", edax_bin.c_str());
  }

  namespace bp = boost::process;
  child_ = new bp::child(edax_relative_bin.c_str(), bp::start_dir(edax_dir), bp::std_out > out_,
                         bp::std_err > bp::null, bp::std_in < in_);
}

EdaxOracle::~EdaxOracle() {
  if (child_) {
    child_->terminate();
    delete child_;
  }
  child_ = nullptr;
}

core::action_t EdaxOracle::query(int depth, const State& state, const ActionMask& valid_actions) {
  // Input:
  // setboard <board_str> <cur_player>\n
  //
  // board_str consists of 64 chars, each "X", "O", and "."
  // cur_player is either "X" or "O"
  //
  // Output:
  /*

 depth|score|       time   |  nodes (N)  |   N/s    | principal variation
------+-----+--------------+-------------+----------+----------------------
    5   +04        0:00.003           799     266333 d3 C3 c4
------+-----+--------------+-------------+----------+----------------------

Edax plays D3

  A B C D E F G H            BLACK            A  B  C  D  E  F  G  H
1 - - - - - - - - 1         0:03.958       1 |  |  |  |  |  |  |  |  | 1
2 - - - - - - - - 2    5 discs   2 moves   2 |  |  |  |  |  |  |  |  | 2
3 - - . * . - - - 3                        3 |  |  |  | 1|  |  |  |  | 3
4 - - - * * - - - 4  ply  2 (58 empties)   4 |  |  |  |()|##|  |  |  | 4
5 - - . * O * . - 5    White's turn (O)    5 |  |  |  |##|()|##|  |  | 5
6 - - - - - - - - 6                        6 |  |  |  |  |  |  |  |  | 6
7 - - - - - - - - 7    1 discs   4 moves   7 |  |  |  |  |  |  |  |  | 7
8 - - - - - - - - 8         0:00.000       8 |  |  |  |  |  |  |  |  | 8
  A B C D E F G H            WHITE            A  B  C  D  E  F  G  H

  */

  // 76 = strlen("setboard") + 1 + 64 + 1 + 1 + strlen("\n")
  constexpr int input_len = 76;
  char input[input_len];
  Game::IO::write_edax_board_str(input, state);
  in_.write(input, input_len);

  // write "level {}" to input:
  int len = sprintf(input, "level %d\n", depth);
  in_.write(input, len);

  int num_hints = deterministic_mode_ ? 1 : 3;
  RELEASE_ASSERT(num_hints > 0 && num_hints <= 9);
  char buf[16] = "hint X\n";
  buf[5] = '0' + num_hints;  // set the number of hints
  in_.write(buf, 7);  // "hint X" means "show top X moves"
  in_.flush();

  /*
  Output:

 depth|score|       time   |  nodes (N)  |   N/s    | principal variation
------+-----+--------------+-------------+----------+----------------------
    5   -06        0:00.001         20967   20967000 c2 F7 d7
    5   -06        0:00.000          3294            c7 F2 d2
    5   -07        0:00.001          6006    6006000 e2 F7 d7
    5   -07        0:00.000          5403            e7 F2 d2
    5   -10        0:00.000          4810            g4 G5 g3
------+-----+--------------+-------------+----------+----------------------

  Output if forced to pass:

 depth|score|       time   |  nodes (N)  |   N/s    | principal variation
------+-----+--------------+-------------+----------+----------------------
------+-----+--------------+-------------+----------+----------------------

  Output if using book:

 depth|score|       time   |  nodes (N)  |   N/s    | principal variation
------+-----+--------------+-------------+----------+----------------------
book    +1                                          d3
------+-----+--------------+-------------+----------+----------------------

  */

  // Parse the above output. Grab all the rows whose score matches the first row's score.
  //
  // In deterministic mode, just return the first move from the first row.
  //
  // Otherwise, choose uniformly randomly among the grabbed moves.
  //
  // If there are no rows in the output, then return kPass.

  int num_candidates = 0;

  bool hit_depth_line = false;
  bool started = false;  // whether we have started parsing the main section
  int best_score = 0;
  int action = kPass;
  size_t line_num = 0;

  auto increment_line_num = [&]() {
    if (line_num + 1 == line_buffer_.size()) {
      line_buffer_.resize(line_buffer_.size() * 2);
    }
    ++line_num;
  };

  for (; std::getline(out_, line_buffer_[line_num]); increment_line_num()) {
    const std::string& line = line_buffer_[line_num];
    if (verbose_) {
      std::cout << line << std::endl;
    }

    if (line.starts_with(" depth")) {
      hit_depth_line = true;
      continue;
    }

    if (!hit_depth_line) {
      // skip lines until we hit the depth line
      continue;
    }

    if (line.starts_with("------+")) {
      if (started) {
        // end of the main section
        break;
      }
      started = true;
      continue;
    }

    if (!started) continue;

    // Example line:
    //     5   -06        0:00.001         20967   20967000 c2 F7 d7
    // Note that the N/s column is optional!

    int n_tokens = util::split(tokens_, line);
    if (n_tokens == 0) continue;

    int move_index = 2;
    bool is_book = false;
    if (n_tokens == 3) {
      RELEASE_ASSERT(util::ends_with(tokens_[0], "book"),
                           "EdaxOracle::query: unexpected line [{}]", line);
      RELEASE_ASSERT(num_candidates == 0, "EdaxOracle::query: {} candidates [{}]",
                           num_candidates, line);
      is_book = true;
    } else {
      RELEASE_ASSERT(n_tokens >= 5, "EdaxOracle::query: got {} tokens [{}]", n_tokens, line);

      int score = std::stoi(tokens_[1]);

      if (num_candidates == 0) {
        best_score = score;
      } else {
        if (score != best_score) {
          // different score, stop parsing
          break;
        }
      }

      num_candidates++;
      if (!deterministic_mode_ && num_candidates > 1) {
        // reservoir sampling: use this candidate with probability 1/num_candidates
        if (util::Random::uniform_sample(0, num_candidates) > 0) continue;
      }

      // if the N/s column is present, column 4 will start with a digit, and the move will be in
      // column 5
      move_index = 4;
      char first_char = tokens_[4][0];
      move_index += std::isdigit(first_char);
    }

    RELEASE_ASSERT(tokens_[move_index].size() == 2,
                         "EdaxOracle::query: tokens_[{}]=\"{}\" [{}]", move_index,
                         tokens_[move_index], line);

    char move_char1 = tokens_[move_index][0];
    char move_char2 = tokens_[move_index][1];

    int row = move_char2 - '1';  // convert '1' to 0, '2' to 1, etc.

    // Note that the output uses lower-case or upper-case depending on the player.
    char base_col_char = (move_char1 >= 'a') ? 'a' : 'A';
    int col = move_char1 - base_col_char;  // convert 'a'/'A' to 0, 'b'/'B' to 1, etc.

    action = col + 8 * row;  // convert to global action index

    if (is_book) {
      // if this is a book move, we can stop parsing
      break;
    }
  }

  return action;
}

}  // namespace othello
