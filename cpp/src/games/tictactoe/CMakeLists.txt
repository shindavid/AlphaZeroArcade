add_executable(tictactoe
        main/TicTacToe.cpp
        GameState.cpp
        PerfectStrategyLookupTable.cpp
        )

set(LOCAL_EXECUTABLES tictactoe)
set(ALL_EXECUTABLES ${ALL_EXECUTABLES} ${LOCAL_EXECUTABLES} PARENT_SCOPE)
