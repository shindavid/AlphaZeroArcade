// cpp/mock_tictactoe.cpp
#include <boost/asio.hpp>

#include <ctime>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using boost::asio::ip::tcp;


static std::mt19937 rng((unsigned)std::time(nullptr));


// Check for a winner or draw. Returns 'X', 'O', 'D' (draw), or '_' (ongoing)
char check_winner(const std::vector<std::string>& board) {
  const int wins[8][3] = {
    {0,1,2},{3,4,5},{6,7,8}, // rows
    {0,3,6},{1,4,7},{2,5,8}, // cols
    {0,4,8},{2,4,6}          // diags
  };
  for (auto& w : wins) {
    if (!board[w[0]].empty() &&
        board[w[0]] == board[w[1]] &&
        board[w[1]] == board[w[2]]) {
      return board[w[0]][0];
    }
  }
  for (int i = 0; i < 9; ++i) if (board[i].empty()) return '_';
  return 'D'; // draw
}

void send_state(tcp::socket& sock, const std::vector<std::string>& board, bool xTurn) {
  // Build a 9‑char board string: 'X', 'O', or '_' for empty
  std::string s;
  s.reserve(9);
  for (int i = 0; i < 9; ++i) {
    if (board[i].empty())
      s.push_back('_');
    else if (board[i] == "X")
      s.push_back('X');
    else if (board[i] == "O")
      s.push_back('O');
    else
      s.push_back('_');  // fallback
  }

  // JSON payload:
  // {"type":"state_update","payload":{"board":"X_O_XO_OX","turn":"X"}}
  std::ostringstream ss;
  ss << "{\"type\":\"state_update\",\"payload\":{"
        "\"board\":\""
     << s
     << "\","
        "\"turn\":\""
     << (xTurn ? "X" : "O")
     << "\""
        "}}\n";

  auto out = ss.str();
  std::cerr << "[engine] Sending state: " << out;
  boost::asio::write(sock, boost::asio::buffer(out));
}

int main(int argc, char* argv[]) {
  int port = 4000;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--port") port = std::stoi(argv[i + 1]);
  }

  std::cerr << "[engine] Starting on port " << port << "\n";

  boost::asio::io_context io_context;
  tcp::acceptor acceptor(io_context);

  // 1) Open the acceptor
  acceptor.open(tcp::v4());
  // 2) Allow immediate reuse of this address
  acceptor.set_option(boost::asio::socket_base::reuse_address(true));
  // 3) Bind and listen
  acceptor.bind(tcp::endpoint(tcp::v4(), port));
  acceptor.listen();

  std::cerr << "[engine] Waiting for connection...\n";

  tcp::socket socket(io_context);
  acceptor.accept(socket);
  std::cerr << "[engine] Client connected\n";

  std::vector<std::string> board(9);
  bool xTurn = true;
  send_state(socket, board, xTurn);

  for (;;) {
    boost::asio::streambuf buf;
    boost::asio::read_until(socket, buf, '\n');
    std::istream is(&buf);
    std::string line;
    std::getline(is, line);
    std::cerr << "[engine] Received: " << line << "\n";

    if (line.find("\"type\":\"make_move\"") != std::string::npos) {
      auto pos = line.find("\"index\":");
      if (pos != std::string::npos) {
        int idx = std::stoi(line.substr(pos + 8));
        if (idx >= 0 && idx < 9 && board[idx].empty()) {
          board[idx] = xTurn ? "X" : "O";
          xTurn = !xTurn;

          // Check for win/draw after player move
          char winner = check_winner(board);
          if (winner == '_' ) {
            // opponent move only if game not over
            std::vector<int> empties;
            for (int j = 0; j < 9; ++j)
              if (board[j].empty()) empties.push_back(j);
            if (!empties.empty()) {
              std::uniform_int_distribution<int> dist(0, empties.size() - 1);
              int c = empties[dist(rng)];
              board[c] = xTurn ? "X" : "O";
              xTurn = !xTurn;
              winner = check_winner(board);
            }
          }

          send_state(socket, board, xTurn);

          // If game ended, send game_end message
          if (winner != '_') {
            std::ostringstream ss;
            ss << "{\"type\":\"game_end\",\"payload\":{";
            if (winner == 'D')
              ss << "\"result\":\"draw\"}}\n";
            else
              ss << "\"result\":\"win\",\"winner\":\"" << winner << "\"}}\n";
            auto out = ss.str();
            std::cerr << "[engine] Sending game_end: " << out;
            boost::asio::write(socket, boost::asio::buffer(out));
          }
        }
      }
    } else if (line.find("\"type\":\"new_game\"") != std::string::npos) {
      // Reset board and turn
      for (auto& s : board) s.clear();
      xTurn = true;
      send_state(socket, board, xTurn);
    }
  }
}
