add_executable(mcts_player_test main/UnitTests.cpp)
target_link_libraries(mcts_player_test PRIVATE ${GTEST_LIBS})

set(LINK_LIBS tictactoe_common_objs util_lib ${COMMON_EXTERNAL_LIBS})
set(TARGETS mcts_player_test)
foreach(target ${TARGETS})
    target_link_libraries(${target} PRIVATE ${LINK_LIBS})
endforeach()

set_target_properties(mcts_player_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")

append_target_metadata("players" ${TARGETS})
