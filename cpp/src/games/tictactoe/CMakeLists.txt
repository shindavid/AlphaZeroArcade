add_executable(tictactoe_exe
        main/TicTacToe.cpp
        Game.cpp
        PerfectStrategyLookupTable.cpp
        )

add_library(tictactoe_ffi
        SHARED shared/TicTacToe_ffi.cpp
        Game.cpp
)

set_target_properties(tictactoe_exe PROPERTIES OUTPUT_NAME "tictactoe")
set_target_properties(tictactoe_ffi PROPERTIES OUTPUT_NAME "tictactoe")

add_custom_target(tictactoe
    DEPENDS tictactoe_exe tictactoe_ffi
)

set(LOCAL_EXECUTABLES tictactoe_exe)
set(ALL_EXECUTABLES ${ALL_EXECUTABLES} ${LOCAL_EXECUTABLES} PARENT_SCOPE)

set(LOCAL_FFI_LIBS tictactoe_ffi)
set(ALL_FFI_LIBS ${ALL_FFI_LIBS} ${LOCAL_FFI_LIBS} PARENT_SCOPE)
