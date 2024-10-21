add_executable(nim_tests main/UnitTests.cpp)
target_link_libraries(nim_tests PRIVATE GTest::GTest GTest::Main)

set(LINK_LIBS util_lib ${TORCH_LIBRARIES} ${Boost_LIBRARIES})
set(TARGETS nim_tests)
foreach(target ${TARGETS})
    target_link_libraries(${target} PRIVATE ${LINK_LIBS})
endforeach()

set_target_properties(nim_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")

append_target_metadata("nim" ${TARGETS})