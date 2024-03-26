add_executable(c4
        main/Connect4.cpp
        GameState.cpp
        PerfectPlayer.cpp
        )

set(LOCAL_EXECUTABLES c4)
set(ALL_EXECUTABLES ${ALL_EXECUTABLES} ${LOCAL_EXECUTABLES} PARENT_SCOPE)
