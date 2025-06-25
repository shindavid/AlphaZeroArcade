set(COMMON_SOURCES
    Game.cpp
)

add_library(hex_common_objs OBJECT ${COMMON_SOURCES})
set_target_properties(hex_common_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(hex_tests main/UnitTests.cpp)
add_executable(hex_exe main/Hex.cpp)
add_library(hex_ffi SHARED shared/Hex_ffi.cpp)
set_target_properties(hex_ffi PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hex_tests PRIVATE ${GTEST_LIBS})

set(LINK_LIBS hex_common_objs util_lib ${COMMON_EXTERNAL_LIBS})
set(TARGETS hex_tests hex_exe hex_ffi)
foreach(target ${TARGETS})
    target_link_libraries(${target} PRIVATE ${LINK_LIBS})
endforeach()

set_target_properties(hex_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")
set_target_properties(hex_exe PROPERTIES OUTPUT_NAME "hex")
set_target_properties(hex_ffi PROPERTIES OUTPUT_NAME "hex")

append_target_metadata("hex" ${TARGETS})
