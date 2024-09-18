set(COMMON_SOURCES
    Game.cpp
    PerfectStrategyLookupTable.cpp
)

add_executable(tictactoe_tests main/UnitTests.cpp ${COMMON_SOURCES})
add_executable(tictactoe_exe main/TicTacToe.cpp ${COMMON_SOURCES})
add_library(tictactoe_ffi SHARED shared/TicTacToe_ffi.cpp ${COMMON_SOURCES})
set_target_properties(tictactoe_ffi PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(LINK_LIBS util_lib ${TORCH_LIBRARIES} ${Boost_LIBRARIES})
set(TARGETS tictactoe_tests tictactoe_exe tictactoe_ffi)
foreach(target ${TARGETS})
    target_link_libraries(${target} PRIVATE ${LINK_LIBS})
endforeach()

set_target_properties(tictactoe_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")
set_target_properties(tictactoe_exe PROPERTIES OUTPUT_NAME "tictactoe")
set_target_properties(tictactoe_ffi PROPERTIES OUTPUT_NAME "tictactoe")

add_custom_target(tictactoe
    DEPENDS tictactoe_exe tictactoe_ffi tictactoe_tests
)
