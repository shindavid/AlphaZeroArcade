set(EXTRA_DEPS_DIR ${CMAKE_SOURCE_DIR}/extra_deps)
set(LC0_DIR ${EXTRA_DEPS_DIR}/lc0)

# --- NEW: Protobuf Setup Start ---
find_package(Protobuf REQUIRED)

# Define where the .proto files are located
set(LC0_COMMON_DIR "${LC0_DIR}/lczero-common")

# Define the output files (generated inside the build directory)
set(PROTO_GEN_DIR "${CMAKE_BINARY_DIR}/proto")
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

set(PROTO_SRCS
    ${PROTO_GEN_DIR}/net.pb.cc
    ${PROTO_GEN_DIR}/chunk.pb.cc
)
set(PROTO_HDRS
    ${PROTO_GEN_DIR}/net.pb.h
    ${PROTO_GEN_DIR}/chunk.pb.h
)

# Run protoc
add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --cpp_out=${CMAKE_BINARY_DIR}
         -I${LC0_COMMON_DIR}
         ${LC0_COMMON_DIR}/proto/net.proto
         ${LC0_COMMON_DIR}/proto/chunk.proto
    DEPENDS ${LC0_COMMON_DIR}/proto/net.proto ${LC0_COMMON_DIR}/proto/chunk.proto
    COMMENT "Generating Lc0 Protobuf C++ files..."
)

# Create a library for the generated proto files
add_library(lc0_proto_lib STATIC ${PROTO_SRCS} ${PROTO_HDRS})

# Link against the system Protobuf library
if(TARGET protobuf::libprotobuf)
    target_link_libraries(lc0_proto_lib PUBLIC protobuf::libprotobuf)
else()
    target_link_libraries(lc0_proto_lib PUBLIC ${Protobuf_LIBRARIES})
    target_include_directories(lc0_proto_lib PUBLIC ${Protobuf_INCLUDE_DIRS})
endif()

# Make sure "proto/net.pb.h" can be included by adding build dir to path
target_include_directories(lc0_proto_lib PUBLIC ${CMAKE_BINARY_DIR})
# --- NEW: Protobuf Setup End ---

set(LC0_SOURCES
    ${LC0_DIR}/chess/board.cc
    ${LC0_DIR}/chess/position.cc
    ${LC0_DIR}/neural/encoder.cc
    ${LC0_DIR}/utils/logging.cc
    ${LC0_DIR}/utils/string.cc
)

add_library(lc0_lib STATIC ${LC0_SOURCES})
target_include_directories(lc0_lib SYSTEM PUBLIC ${LC0_DIR})
set_target_properties(lc0_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(lc0_lib PRIVATE -mbmi2 -w)

target_link_libraries(lc0_lib PUBLIC lc0_proto_lib)

set(COMMON_SOURCES
    Game.cpp
)

include_directories(SYSTEM ${LC0_DIR})

set(EXTRA_INCLUDES
    ${EXTRA_DEPS_DIR}  # for our usage of lc0
    ${CMAKE_BINARY_DIR}
)

add_game(
    NAME chess
    EXE main/Chess.cpp
    TEST main/UnitTests.cpp
    FFI shared/Chess_ffi.cpp
    SOURCES ${COMMON_SOURCES}
    INCLUDES ${EXTRA_INCLUDES}
)

# Hook lc0_lib into the chess targets
foreach(tgt IN ITEMS chess_tests chess_exe chess_ffi)
  target_link_libraries(${tgt} PRIVATE lc0_lib)
  set_target_properties(${tgt} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endforeach()
