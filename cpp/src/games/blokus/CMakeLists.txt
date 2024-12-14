set(COMMON_SOURCES
    Game.cpp
    GameState.cpp
    Pieces.cpp
    Types.cpp
)

add_library(blokus_common_objs OBJECT ${COMMON_SOURCES})
set_target_properties(blokus_common_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(blokus_tests main/UnitTests.cpp)
add_executable(blokus_exe main/Blokus.cpp)
add_library(blokus_ffi SHARED shared/Blokus_ffi.cpp)
set_target_properties(blokus_ffi PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(blokus_tests PRIVATE ${GTEST_LIBS})

set(LINK_LIBS blokus_common_objs util_lib ${COMMON_EXTERNAL_LIBS})
set(TARGETS blokus_tests blokus_exe blokus_ffi)
foreach(target ${TARGETS})
    target_link_libraries(${target} PRIVATE ${LINK_LIBS})
endforeach()

set_target_properties(blokus_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")
set_target_properties(blokus_exe PROPERTIES OUTPUT_NAME "blokus")
set_target_properties(blokus_ffi PROPERTIES OUTPUT_NAME "blokus")

append_target_metadata("blokus" ${TARGETS})
