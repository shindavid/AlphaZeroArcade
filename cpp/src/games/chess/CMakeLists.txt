set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mbmi2")

include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/extra_deps)

# Fathom Syzygy tablebase probing library (C code).
# Our local tbconfig.h is picked up via include path priority over Fathom's default.
set(FATHOM_DIR ${CMAKE_SOURCE_DIR}/extra_deps/Fathom/src)
include_directories(SYSTEM ${FATHOM_DIR})
add_library(fathom STATIC ${FATHOM_DIR}/tbprobe.c)
target_include_directories(fathom PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${FATHOM_DIR})
set_target_properties(fathom PROPERTIES C_STANDARD 11 POSITION_INDEPENDENT_CODE ON)

set(COMMON_SOURCES
    Game.cpp
    PolicyEncoding.cpp
)

add_game(
    NAME chess
    EXE main/Chess.cpp
    TEST main/UnitTests.cpp
    FFI shared/Chess_ffi.cpp
    SOURCES ${COMMON_SOURCES}
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
    EXTRA_LIBS fathom
)
