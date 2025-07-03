set(COMMON_SOURCES
    Game.cpp
    GameState.cpp
    Pieces.cpp
    Types.cpp
)

add_game(
    NAME blokus
    EXE main/Blokus.cpp
    TEST main/UnitTests.cpp
    FFI shared/Blokus_ffi.cpp
    SOURCES ${COMMON_SOURCES}
)
