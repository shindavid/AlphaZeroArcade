#include <core/Main.hpp>

#include <games/chess/PlayerFactory.hpp>

int main(int ac, char* av[]) { return Main<chess::PlayerFactory>::main(ac, av); }
