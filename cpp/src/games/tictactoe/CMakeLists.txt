set(COMMON_SOURCES
    Game.cpp
    PerfectStrategyLookupTable.cpp
)

add_library(tictactoe_common_objs OBJECT ${COMMON_SOURCES})
set_target_properties(tictactoe_common_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(tictactoe_tests main/UnitTests.cpp)
add_executable(tictactoe_exe main/TicTacToe.cpp)
add_library(tictactoe_ffi SHARED shared/TicTacToe_ffi.cpp)
set_target_properties(tictactoe_ffi PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(tictactoe_tests PRIVATE ${GTEST_LIBS})

set(LINK_LIBS tictactoe_common_objs util_lib ${COMMON_EXTERNAL_LIBS})
set(TARGETS tictactoe_tests tictactoe_exe tictactoe_ffi)
foreach(target ${TARGETS})
    target_link_libraries(${target} PRIVATE ${LINK_LIBS})
endforeach()

set_target_properties(tictactoe_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")
set_target_properties(tictactoe_exe PROPERTIES OUTPUT_NAME "tictactoe")
set_target_properties(tictactoe_ffi PROPERTIES OUTPUT_NAME "tictactoe")

append_target_metadata("tictactoe" ${TARGETS})
