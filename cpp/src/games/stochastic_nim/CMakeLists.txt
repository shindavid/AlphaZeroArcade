# set(COMMON_SOURCES
# )
# add_library(c4_common_objs OBJECT ${COMMON_SOURCES})
# set_target_properties(stochastic_nim_common_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(stochastic_nim_tests main/UnitTests.cpp)
add_executable(stochastic_nim_exe main/stochastic_nim.cpp)
add_library(stochastic_nim_ffi SHARED shared/stochastic_nim_ffi.cpp)
set_target_properties(stochastic_nim_ffi PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(stochastic_nim_tests PRIVATE ${GTEST_LIBS})

set(LINK_LIBS util_lib ${COMMON_EXTERNAL_LIBS})
set(TARGETS stochastic_nim_tests stochastic_nim_exe stochastic_nim_ffi)
foreach(target ${TARGETS})
    target_link_libraries(${target} PRIVATE ${LINK_LIBS})
endforeach()

set_target_properties(stochastic_nim_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")
set_target_properties(stochastic_nim_exe PROPERTIES OUTPUT_NAME "stochastic_nim")
set_target_properties(stochastic_nim_ffi PROPERTIES OUTPUT_NAME "stochastic_nim")

append_target_metadata("stochastic_nim" ${TARGETS})