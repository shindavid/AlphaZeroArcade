set(COMMON_SOURCES
    Game.cpp
    PerfectPlayer.cpp
)

add_executable(c4_tests main/UnitTests.cpp ${COMMON_SOURCES})
add_executable(c4_exe main/Connect4.cpp ${COMMON_SOURCES})
add_library(c4_ffi SHARED shared/Connect4_ffi.cpp ${COMMON_SOURCES})
set_target_properties(c4_ffi PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(LINK_LIBS util_lib ${TORCH_LIBRARIES} ${Boost_LIBRARIES})
set(TARGETS c4_tests c4_exe c4_ffi)
foreach(target ${TARGETS})
    target_link_libraries(${target} PRIVATE ${LINK_LIBS})
endforeach()

set_target_properties(c4_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")
set_target_properties(c4_exe PROPERTIES OUTPUT_NAME "c4")
set_target_properties(c4_ffi PROPERTIES OUTPUT_NAME "c4")

add_custom_target(c4
    DEPENDS c4_exe c4_ffi c4_tests
)
