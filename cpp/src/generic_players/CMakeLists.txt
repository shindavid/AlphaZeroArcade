add_executable(mcts_player_test main/UnitTests.cpp)

set(TEST_LINK_LIBS tictactoe_common_objs_test util_lib_test)

set(TARGETS mcts_player_test)
foreach(target ${TARGETS})
    target_compile_definitions(${target} PRIVATE ${TEST_COMPILE_DEFINITIONS})
    target_link_libraries(${target} PRIVATE ${TEST_LINK_LIBS})
    target_link_libraries(${target} PRIVATE ${COMMON_EXTERNAL_LIBS} ${GTEST_LIBS})
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")
endforeach()

append_target_metadata("aux" ${TARGETS})
