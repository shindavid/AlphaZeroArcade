#include <common/Main.hpp>
#include <games/tictactoe/PlayerFactory.hpp>

int main(int ac, char* av[]) {
  return Main<tictactoe::PlayerFactory>::main(ac, av);
}
