add_executable(othello_tests
        main/UnitTests.cpp
        Game.cpp
        flip_sse.cpp
        )

 add_executable(othello_exe
        main/Othello.cpp
        Game.cpp
        flip_sse.cpp
        )

add_library(othello_ffi
        SHARED shared/Othello_ffi.cpp
        Game.cpp
        flip_sse.cpp
)

set_target_properties(othello_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")
set_target_properties(othello_exe PROPERTIES OUTPUT_NAME "othello")
set_target_properties(othello_ffi PROPERTIES OUTPUT_NAME "othello")

add_custom_target(othello
    DEPENDS othello_exe othello_ffi othello_tests
)

set(LOCAL_EXECUTABLES othello_exe othello_tests)
set(ALL_EXECUTABLES ${ALL_EXECUTABLES} ${LOCAL_EXECUTABLES} PARENT_SCOPE)

set(LOCAL_FFI_LIBS othello_ffi)
set(ALL_FFI_LIBS ${ALL_FFI_LIBS} ${LOCAL_FFI_LIBS} PARENT_SCOPE)
