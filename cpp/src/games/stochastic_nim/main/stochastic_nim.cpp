#include "core/Main.hpp"
#include "games/stochastic_nim/PlayerFactory.hpp"

int main(int ac, char* av[]) { return Main<stochastic_nim::PlayerFactory>::main(ac, av); }
