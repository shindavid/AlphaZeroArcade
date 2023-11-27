#include <core/Main.hpp>
#include <games/blokus/PlayerFactory.hpp>

int main(int ac, char* av[]) { return Main<blokus::PlayerFactory>::main(ac, av); }
