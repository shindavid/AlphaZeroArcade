set(COMMON_SOURCES
    Game.cpp
    flip_sse.cpp
)

add_executable(othello_tests main/UnitTests.cpp ${COMMON_SOURCES})
add_executable(othello_exe main/Othello.cpp ${COMMON_SOURCES})
add_library(othello_ffi SHARED shared/Othello_ffi.cpp ${COMMON_SOURCES})
set_target_properties(othello_ffi PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(LINK_LIBS util_lib ${TORCH_LIBRARIES} ${Boost_LIBRARIES})
set(TARGETS othello_tests othello_exe othello_ffi)
foreach(target ${TARGETS})
    target_link_libraries(${target} PRIVATE ${LINK_LIBS})
endforeach()

set_target_properties(othello_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")
set_target_properties(othello_exe PROPERTIES OUTPUT_NAME "othello")
set_target_properties(othello_ffi PROPERTIES OUTPUT_NAME "othello")

append_target_metadata("othello" ${TARGETS})
