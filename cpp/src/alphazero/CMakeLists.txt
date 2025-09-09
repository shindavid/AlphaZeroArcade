add_test(
    NAME alphazero_tests
    TEST main/UnitTests.cpp
    EXTRA_LIBS tictactoe_common_objs_test util_lib_test
)

add_test(
    NAME gameserver_tests
    TEST main/GameServerUnitTests.cpp
    EXTRA_LIBS tictactoe_common_objs_test othello_common_objs_test util_lib_test
)
