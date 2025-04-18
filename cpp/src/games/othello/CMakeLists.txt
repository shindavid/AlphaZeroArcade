set(COMMON_SOURCES
    EdaxOracle.cpp
    Game.cpp
    flip_sse.cpp
)

add_library(othello_common_objs OBJECT ${COMMON_SOURCES})
set_target_properties(othello_common_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(othello_tests main/UnitTests.cpp)
add_executable(othello_exe main/Othello.cpp)
add_library(othello_ffi SHARED shared/Othello_ffi.cpp)
set_target_properties(othello_ffi PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(othello_tests PRIVATE ${GTEST_LIBS})

set(LINK_LIBS othello_common_objs util_lib ${COMMON_EXTERNAL_LIBS})
set(TARGETS othello_tests othello_exe othello_ffi)
foreach(target ${TARGETS})
    target_link_libraries(${target} PRIVATE ${LINK_LIBS})
endforeach()

set_target_properties(othello_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")
set_target_properties(othello_exe PROPERTIES OUTPUT_NAME "othello")
set_target_properties(othello_ffi PROPERTIES OUTPUT_NAME "othello")

append_target_metadata("othello" ${TARGETS})
