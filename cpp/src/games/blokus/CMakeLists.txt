add_executable(blokus_tests
        main/UnitTests.cpp
        Pieces.cpp
        )

add_executable(blokus_exe
        main/Blokus.cpp
        Pieces.cpp
        )

add_library(blokus_ffi
        SHARED shared/Blokus_ffi.cpp
        Pieces.cpp
)

set_target_properties(blokus_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")
set_target_properties(blokus_exe PROPERTIES OUTPUT_NAME "blokus")
set_target_properties(blokus_ffi PROPERTIES OUTPUT_NAME "blokus")

add_custom_target(blokus
    DEPENDS blokus_exe blokus_ffi blokus_tests
)

set(LOCAL_EXECUTABLES blokus_exe blokus_tests)
set(ALL_EXECUTABLES ${ALL_EXECUTABLES} ${LOCAL_EXECUTABLES} PARENT_SCOPE)

set(LOCAL_FFI_LIBS blokus_ffi)
set(ALL_FFI_LIBS ${ALL_FFI_LIBS} ${LOCAL_FFI_LIBS} PARENT_SCOPE)
