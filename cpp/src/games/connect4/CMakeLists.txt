set(COMMON_SOURCES
    Game.cpp
    PerfectOracle.cpp
    PerfectPlayer.cpp
)

    EXE main/Connect4.cpp
    TEST main/UnitTests.cpp
    FFI shared/Connect4_ffi.cpp
    SOURCES ${COMMON_SOURCES}
)
