set(COMMON_SOURCES
    EdaxOracle.cpp
    Game.cpp
    GameState.cpp
    flip_sse.cpp
)

add_game(
    NAME othello
    EXE main/Othello.cpp
    TEST main/UnitTests.cpp
    FFI shared/Othello_ffi.cpp
    SOURCES ${COMMON_SOURCES}
)
