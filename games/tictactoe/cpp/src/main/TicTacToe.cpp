#include <core/Main.hpp>
#include <tictactoe/PlayerFactory.hpp>

int main(int ac, char* av[]) { return Main<tictactoe::PlayerFactory>::main(ac, av); }
