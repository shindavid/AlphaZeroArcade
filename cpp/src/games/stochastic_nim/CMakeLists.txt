add_executable(stochastic_nim_tests main/UnitTests.cpp)
add_executable(stochastic_nim_perfect_player_tests PerfectPlayerUnitTests.cpp)

target_link_libraries(stochastic_nim_tests PRIVATE ${GTEST_LIBS})
target_link_libraries(stochastic_nim_perfect_player_tests PRIVATE ${GTEST_LIBS})

set(LINK_LIBS util_lib ${COMMON_EXTERNAL_LIBS})
set(TARGETS stochastic_nim_tests stochastic_nim_perfect_player_tests)
foreach(target ${TARGETS})
    target_link_libraries(${target} PRIVATE ${LINK_LIBS})
endforeach()

set_target_properties(stochastic_nim_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")
set_target_properties(stochastic_nim_perfect_player_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")

append_target_metadata("stochastic_nim" ${TARGETS})