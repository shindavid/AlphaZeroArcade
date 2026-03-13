add_test(
    NAME alpha0_tests
    TEST main/Alpha0UnitTests.cpp
    EXTRA_LIBS tictactoe_common_objs_test core_lib_test
)

add_test(
    NAME gameserver_tests
    TEST main/GameServerUnitTests.cpp
    EXTRA_LIBS tictactoe_common_objs_test core_lib_test
)

add_test(
    NAME mcts_player_test
    TEST main/GenericPlayersUnitTests.cpp
    EXTRA_LIBS tictactoe_common_objs_test core_lib_test
)

add_test(
    NAME remote_player_tests
    TEST main/RemotePlayerUnitTests.cpp
    NO_MIT
    EXTRA_LIBS tictactoe_common_objs c4_common_objs core_lib
)
