set(COMMON_SOURCES
    Game.cpp
)

add_game(
    NAME hex
    EXE main/Hex.cpp
    TEST main/UnitTests.cpp
    FFI shared/Hex_ffi.cpp
    SOURCES ${COMMON_SOURCES}
)
