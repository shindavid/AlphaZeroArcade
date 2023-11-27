#include <core/Main.hpp>
#include <games/othello/PlayerFactory.hpp>

int main(int ac, char* av[]) { return Main<othello::PlayerFactory>::main(ac, av); }
