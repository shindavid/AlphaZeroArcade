set(EXTRA_DEPS_DIR ${CMAKE_SOURCE_DIR}/extra_deps)
set(LC0_DIR ${EXTRA_DEPS_DIR}/lc0)

# file(GLOB LC0_SOURCES ${LC0_DIR}/*/*.cc)
# list(REMOVE_ITEM LC0_SOURCES ${LC0_DIR}/chess/uciloop.cc)
# list(REMOVE_ITEM LC0_SOURCES ${LC0_DIR}/utils/filesystem.win32.cc)
# list(REMOVE_ITEM LC0_SOURCES ${LC0_DIR}/utils/weights_adapter.cc)

set(LC0_SOURCES
    ${LC0_DIR}/chess/bitboard.cc
    ${LC0_DIR}/chess/board.cc
    ${LC0_DIR}/chess/position.cc
    ${LC0_DIR}/utils/logging.cc
    ${LC0_DIR}/utils/string.cc
)

add_library(lc0_lib STATIC ${LC0_SOURCES})
target_include_directories(lc0_lib
    PUBLIC ${LC0_DIR}  # for includes within lc0
)
set_target_properties(lc0_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(lc0_lib PRIVATE -mbmi2 -Wno-restrict)

add_library(chess_lib STATIC Game.cpp)

set_target_properties(chess_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(chess_lib
    PUBLIC ${EXTRA_DEPS_DIR}  # for our usage of lc0
    PUBLIC ${LC0_DIR}  # for includes within lc0
)

add_executable(chess_tests main/UnitTests.cpp)
add_executable(chess_exe main/Chess.cpp)
add_library(chess_ffi SHARED shared/Chess_ffi.cpp)
set_target_properties(chess_ffi PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(chess_tests PRIVATE ${GTEST_LIBS})

set(LINK_LIBS lc0_lib chess_lib util_lib ${COMMON_EXTERNAL_LIBS})
set(TARGETS chess_tests chess_exe chess_ffi)
foreach(target ${TARGETS})
    target_link_libraries(${target} PRIVATE ${LINK_LIBS})
    set_target_properties(${target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endforeach()

set_target_properties(chess_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")
set_target_properties(chess_exe PROPERTIES OUTPUT_NAME "chess")
set_target_properties(chess_ffi PROPERTIES OUTPUT_NAME "chess")

append_target_metadata("chess" ${TARGETS})