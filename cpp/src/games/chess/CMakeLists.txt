set(EXTRA_DEPS_DIR ${CMAKE_SOURCE_DIR}/extra_deps)
set(LC0_DIR ${EXTRA_DEPS_DIR}/lc0)

set(LC0_SOURCES
    ${LC0_DIR}/chess/board.cc
    ${LC0_DIR}/chess/position.cc
    ${LC0_DIR}/utils/logging.cc
    ${LC0_DIR}/utils/string.cc
)

add_library(lc0_lib STATIC ${LC0_SOURCES})
target_include_directories(lc0_lib SYSTEM PUBLIC ${LC0_DIR})
set_target_properties(lc0_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(lc0_lib PRIVATE -mbmi2 -w)

set(COMMON_SOURCES
    Game.cpp
)

include_directories(SYSTEM ${LC0_DIR})

set(EXTRA_INCLUDES
    ${EXTRA_DEPS_DIR}  # for our usage of lc0
)

add_game(
    NAME chess
    EXE main/Chess.cpp
    TEST main/UnitTests.cpp
    FFI shared/Chess_ffi.cpp
    SOURCES ${COMMON_SOURCES}
    INCLUDES ${EXTRA_INCLUDES}
)

# Hook lc0_lib into the chess targets
foreach(tgt IN ITEMS chess_tests chess_exe chess_ffi)
  target_link_libraries(${tgt} PRIVATE lc0_lib)
  set_target_properties(${tgt} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endforeach()
