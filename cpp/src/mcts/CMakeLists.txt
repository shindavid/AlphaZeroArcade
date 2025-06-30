add_executable(mcts_tests main/UnitTests.cpp)
add_executable(gameserver_tests main/GameServerUnitTests.cpp)

set(LINK_LIBS tictactoe_common_objs othello_common_objs util_lib ${COMMON_EXTERNAL_LIBS})

set(TARGETS mcts_tests gameserver_tests)
foreach(target ${TARGETS})
    target_link_libraries(${target} PRIVATE ${LINK_LIBS})
    target_link_libraries(${target} PRIVATE ${GTEST_LIBS})
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")
endforeach()

append_target_metadata("mcts" ${TARGETS})
