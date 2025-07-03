add_game(
    NAME stochastic_nim
    EXE main/stochastic_nim.cpp
    TEST main/UnitTests.cpp
    FFI shared/stochastic_nim_ffi.cpp
)
