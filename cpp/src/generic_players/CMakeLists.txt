add_test(
    NAME mcts_player_test
    TEST main/UnitTests.cpp
    EXTRA_LIBS tictactoe_common_objs_test util_lib_test
)
