add_executable(c4_exe
        main/Connect4.cpp
        Game.cpp
        PerfectPlayer.cpp
        )

add_library(c4_ffi
        SHARED shared/Connect4_ffi.cpp
)

set_target_properties(c4_exe PROPERTIES OUTPUT_NAME "c4")
set_target_properties(c4_ffi PROPERTIES OUTPUT_NAME "c4")

add_custom_target(c4
    DEPENDS c4_exe c4_ffi
)

set(LOCAL_EXECUTABLES c4_exe)
set(ALL_EXECUTABLES ${ALL_EXECUTABLES} ${LOCAL_EXECUTABLES} PARENT_SCOPE)

set(LOCAL_FFI_LIBS c4_ffi)
set(ALL_FFI_LIBS ${ALL_FFI_LIBS} ${LOCAL_FFI_LIBS} PARENT_SCOPE)
