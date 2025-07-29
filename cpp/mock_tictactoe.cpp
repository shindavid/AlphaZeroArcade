
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

char check_winner(const std::vector<std::string>& board) {
  const int wins[8][3] = {
    {0,1,2},{3,4,5},{6,7,8},
    {0,3,6},{1,4,7},{2,5,8},
    {0,4,8},{2,4,6}
  };
  for (auto& w : wins) {
    if (!board[w[0]].empty() && board[w[0]] == board[w[1]] && board[w[1]] == board[w[2]])
      return board[w[0]][0];
  }
  for (int i = 0; i < 9; ++i) if (board[i].empty()) return '_';
  return 'D';
}

int get_ai_move(const std::vector<std::string>& board) {
  std::vector<int> empties;
  for (int j = 0; j < 9; ++j)
    if (board[j].empty()) empties.push_back(j);
  if (empties.empty()) return -1;
  std::uniform_int_distribution<int> dist(0, empties.size() - 1);
  return empties[dist(rng)];
}

int get_human_move(tcp::socket& socket, bool& new_game) {
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
  } else if (line.find("\"type\":\"new_game\"") != std::string::npos) {
    new_game = true;
    return -1;
  }
  return -1;
}

void apply_move(std::vector<std::string>& board, bool& xTurn, int move) {
  if (move >= 0 && move < 9 && board[move].empty()) {
    board[move] = xTurn ? "X" : "O";
    xTurn = !xTurn;
  }
}

void send_state(tcp::socket& sock, const std::vector<std::string>& board, bool xTurn) {
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
      s.push_back('_');
  }
  std::ostringstream ss;
  ss << "{\"type\":\"state_update\",\"payload\":{"
        "\"board\":\"" << s << "\"," << "\"turn\":\"" << (xTurn ? "X" : "O") << "\"}}\n";
  auto out = ss.str();
  std::cerr << "[engine] Sending state: " << out;
  boost::asio::write(sock, boost::asio::buffer(out));
}

void send_seat_assignment(tcp::socket& sock, bool humanIsX) {
  std::ostringstream ss;
  ss << "{\"type\":\"seat_assignment\",\"payload\":{\"human\":\"" << (humanIsX ? "X" : "O") << "\"}}\n";
  boost::asio::write(sock, boost::asio::buffer(ss.str()));
}

void reset_game(std::vector<std::string>& board, bool& xTurn, bool& humanIsX, tcp::socket& socket) {
  for (auto& s : board) s.clear();
  xTurn = true;
  humanIsX = !humanIsX;
  send_state(socket, board, xTurn);
  send_seat_assignment(socket, humanIsX);
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

  std::vector<std::string> board(9);
  bool xTurn = true;
  static bool humanIsX = std::uniform_int_distribution<int>(0,1)(rng) == 1;

  send_state(socket, board, xTurn);
  send_seat_assignment(socket, humanIsX);

  for (;;) {
    int move = -1;
    bool new_game = false;
    bool humanTurn = (xTurn && humanIsX) || (!xTurn && !humanIsX);
    if (!humanTurn) {
      move = get_ai_move(board);
    } else {
      move = get_human_move(socket, new_game);
    }
    if (new_game) {
      reset_game(board, xTurn, humanIsX, socket);
      continue;
    }
    apply_move(board, xTurn, move);
    send_state(socket, board, xTurn);
    char winner = check_winner(board);
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
