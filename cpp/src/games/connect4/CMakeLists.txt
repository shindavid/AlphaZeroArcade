set(COMMON_SOURCES
    Game.cpp
    PerfectOracle.cpp
    PerfectPlayer.cpp
)

add_library(c4_common_objs OBJECT ${COMMON_SOURCES})
set_target_properties(c4_common_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(c4_tests main/UnitTests.cpp)
add_executable(c4_exe main/Connect4.cpp)
add_library(c4_ffi SHARED shared/Connect4_ffi.cpp)
set_target_properties(c4_ffi PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(c4_tests PRIVATE ${GTEST_LIBS})

set(LINK_LIBS c4_common_objs util_lib ${COMMON_EXTERNAL_LIBS})
set(TARGETS c4_tests c4_exe c4_ffi)
foreach(target ${TARGETS})
    target_link_libraries(${target} PRIVATE ${LINK_LIBS})
endforeach()

set_target_properties(c4_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")
set_target_properties(c4_exe PROPERTIES OUTPUT_NAME "c4")
set_target_properties(c4_ffi PROPERTIES OUTPUT_NAME "c4")

append_target_metadata("c4" ${TARGETS})
