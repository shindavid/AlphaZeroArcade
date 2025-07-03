set(COMMON_SOURCES
    Game.cpp
    PerfectStrategyLookupTable.cpp
)

add_game(
    NAME tictactoe
    EXE main/TicTacToe.cpp
    TEST main/UnitTests.cpp
    FFI shared/TicTacToe_ffi.cpp
    SOURCES ${COMMON_SOURCES}
)
