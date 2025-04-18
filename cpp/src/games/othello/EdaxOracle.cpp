#include <games/othello/EdaxOracle.hpp>

#include <format>

namespace othello {

EdaxOracle::EdaxOracle(int depth, bool verbose) : verbose_(verbose) {
  line_buffer_.resize(1);

  auto edax_dir = boost::filesystem::path("extra_deps/edax-reversi");
  auto edax_relative_bin = boost::filesystem::path("bin/lEdax-x64-modern");
  auto edax_bin = edax_dir / edax_relative_bin;

  if (!boost::filesystem::is_regular_file(edax_bin)) {
    throw util::CleanException("File does not exist: %s", edax_bin.c_str());
  }

  namespace bp = boost::process;
  child_ = new bp::child(edax_relative_bin.c_str(), bp::start_dir(edax_dir), bp::std_out > out_,
                         bp::std_err > bp::null, bp::std_in < in_);

  std::string level_str = std::format("level {}\n", depth);
  in_.write(level_str.c_str(), level_str.size());
  in_.flush();
}

EdaxOracle::~EdaxOracle() {
  if (child_) {
    child_->terminate();
    delete child_;
  }
  child_ = nullptr;
}

core::action_t EdaxOracle::query(const State& state, const ActionMask& valid_actions) {
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
  in_.write("go\n", 3);
  in_.flush();

  int a = -1;
  size_t n = 0;
  for (; std::getline(out_, line_buffer_[n]); ++n) {
    const std::string& line = line_buffer_[n];
    if (line.starts_with("Edax plays ")) {
      if (verbose_) {
        std::cout << line << std::endl;
      }
      std::string move_str = line.substr(11, 2);
      if (move_str.starts_with("PS")) {  // "PS" is edax notation for pass
        a = kPass;
        break;
      } else {
        a = (move_str[0] - 'A') + 8 * (move_str[1] - '1');
        break;
      }
    }
    if (n + 1 == line_buffer_.size()) {
      line_buffer_.resize(line_buffer_.size() * 2);
    }
  }

  if (a < 0 || a >= kNumGlobalActions || !valid_actions[a]) {
    for (size_t i = 0; i < n; ++i) {
      std::cerr << line_buffer_[i] << std::endl;
    }
    throw util::Exception("EdaxPlayer::get_action_response: invalid action: %d", a);
  }

  return a;
}

}  // namespace othello
