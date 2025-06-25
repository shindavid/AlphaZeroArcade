#include <core/Main.hpp>
#include <games/hex/PlayerFactory.hpp>

int main(int ac, char* av[]) { return Main<hex::PlayerFactory>::main(ac, av); }
