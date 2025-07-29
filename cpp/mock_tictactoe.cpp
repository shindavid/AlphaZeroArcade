
// cpp/mock_tictactoe.cpp
#include <boost/asio.hpp>

#include <array>
#include <ctime>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

using boost::asio::ip::tcp;
static std::mt19937 rng((unsigned)std::time(nullptr));

class GameState {
 public:
  GameState()
      : board{'_', '_', '_', '_', '_', '_', '_', '_', '_'},
        xTurn(true),
        humanIsX(std::uniform_int_distribution<int>(0, 1)(rng) == 1) {}

  char check_winner() const {
    const int wins[8][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6},
                            {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6}};
    for (auto& w : wins) {
      if (board[w[0]] != '_' && board[w[0]] == board[w[1]] && board[w[1]] == board[w[2]])
        return board[w[0]];
    }
    for (int i = 0; i < 9; ++i)
      if (board[i] == '_') return '_';
    return 'D';
  }

  int get_ai_move() const {
    std::vector<int> empties;
    for (int j = 0; j < 9; ++j)
      if (board[j] == '_') empties.push_back(j);
    if (empties.empty()) return -1;
    std::uniform_int_distribution<int> dist(0, empties.size() - 1);
    return empties[dist(rng)];
  }

  int get_human_move(tcp::socket& socket) const {
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
        return idx;
      }
    }
    return -1;
  }

  void apply_move(int move) {
    if (move >= 0 && move < 9 && board[move] == '_') {
      board[move] = xTurn ? 'X' : 'O';
      xTurn = !xTurn;
    }
  }

  void send_state(tcp::socket& sock) const {
    std::string s(board.begin(), board.end());
    std::ostringstream ss;
    ss << "{\"type\":\"state_update\",\"payload\":{";
    ss << "\"board\":\"" << s << "\",";
    ss << "\"turn\":\"" << (xTurn ? 'X' : 'O') << "\"}}\n";
    auto out = ss.str();
    std::cerr << "[engine] Sending state: " << out;
    boost::asio::write(sock, boost::asio::buffer(out));
  }

  void send_seat_assignment(tcp::socket& sock) const {
    std::ostringstream ss;
    ss << "{\"type\":\"seat_assignment\",\"payload\":{\"human\":\"" << (humanIsX ? 'X' : 'O')
       << "\"}}\n";
    boost::asio::write(sock, boost::asio::buffer(ss.str()));
  }

  void reset(tcp::socket& socket) {
    board.fill('_');
    xTurn = true;
    humanIsX = !humanIsX;
    send_state(socket);
    send_seat_assignment(socket);
  }

  bool is_human_turn() const { return (xTurn && humanIsX) || (!xTurn && !humanIsX); }

 private:
  std::array<char, 9> board;
  bool xTurn;
  bool humanIsX;
};

void send_game_end(tcp::socket& socket, char winner) {
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

void await_new_game_cmd(tcp::socket& socket, GameState& state) {
  while (true) {
    boost::asio::streambuf buf;
    boost::asio::read_until(socket, buf, '\n');
    std::istream is(&buf);
    std::string line;
    std::getline(is, line);
    std::cerr << "[engine] Waiting for new_game, received: " << line << "\n";
    if (line.find("\"type\":\"new_game\"") != std::string::npos) {
      state.reset(socket);
      break;
    }
  }
}

int main(int argc, char* argv[]) {
  int port = 4000;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--port") port = std::stoi(argv[i + 1]);
  }

  std::cerr << "[engine] Starting on port " << port << "\n";

  boost::asio::io_context io_context;
  tcp::acceptor acceptor(io_context);
  acceptor.open(tcp::v4());
  acceptor.set_option(boost::asio::socket_base::reuse_address(true));
  acceptor.bind(tcp::endpoint(tcp::v4(), port));
  acceptor.listen();

  std::cerr << "[engine] Waiting for connection...\n";
  tcp::socket socket(io_context);
  acceptor.accept(socket);
  std::cerr << "[engine] Client connected\n";

  GameState state;

  for (;;) {  // outer loop: new games
    state.send_state(socket);
    state.send_seat_assignment(socket);

    while (true) {  // inner loop: turns within a game
      int move = -1;
      if (!state.is_human_turn()) {
        move = state.get_ai_move();
      } else {
        move = state.get_human_move(socket);
      }
      state.apply_move(move);
      state.send_state(socket);
      char winner = state.check_winner();
      if (winner != '_') {
        send_game_end(socket, winner);
        break;  // break inner loop, wait for new game
      }
    }
    await_new_game_cmd(socket, state);
  }
}
