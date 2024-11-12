# this is a temporary hack until David restructures the cmake files

set(COMMON_SOURCES
    ../games/tictactoe/Game.cpp
    ../games/tictactoe/PerfectStrategyLookupTable.cpp
)

# Define the unit test target and include common sources
add_executable(mcts_tests main/UnitTests.cpp ${COMMON_SOURCES})

# Link Google Test libraries to the unit test executable
target_link_libraries(mcts_tests PRIVATE GTest::GTest GTest::Main)

# Specify additional libraries (util_lib, TORCH_LIBRARIES, Boost) for all targets
set(LINK_LIBS util_lib ${TORCH_LIBRARIES} ${Boost_LIBRARIES})

# Apply the LINK_LIBS to all targets and include necessary directories
set(TARGETS mcts_tests)
foreach(target ${TARGETS})
    target_link_libraries(${target} PRIVATE ${LINK_LIBS})
endforeach()

# Set the output directory for the test executable
set_target_properties(mcts_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")

# Append metadata for the executable
append_target_metadata("mcts" ${TARGETS})