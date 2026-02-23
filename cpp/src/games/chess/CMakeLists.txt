set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mbmi2")

set(COMMON_SOURCES
    Game.cpp
)

add_game(
    NAME chess
    EXE main/Chess.cpp
    TEST main/UnitTests.cpp
    FFI shared/Chess_ffi.cpp
    SOURCES ${COMMON_SOURCES}
)