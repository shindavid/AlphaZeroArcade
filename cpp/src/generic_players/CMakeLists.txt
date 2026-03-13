add_test(
    NAME mcts_player_test
    TEST main/UnitTests.cpp
    EXTRA_LIBS tictactoe_common_objs_test core_lib_test
)

add_test(
    NAME remote_player_tests
    TEST main/RemotePlayerUnitTests.cpp
    NO_MIT
    EXTRA_LIBS tictactoe_common_objs c4_common_objs core_lib
)
