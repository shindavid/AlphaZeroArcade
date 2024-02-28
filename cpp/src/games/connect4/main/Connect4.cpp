#include <core/Main.hpp>

#include <games/connect4/PlayerFactory.hpp>

int main(int ac, char* av[]) {
  return Main<c4::PlayerFactory>::main(ac, av);
}
