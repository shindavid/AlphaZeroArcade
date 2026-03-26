cmake_minimum_required(VERSION 3.17)

set(MAGIC_ENUM_OPT_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(MAGIC_ENUM_OPT_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(MAGIC_ENUM_OPT_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MAGIC_ENUM_OPT_BUILD_DOCS       OFF CACHE BOOL "" FORCE)
include(FetchContent)
FetchContent_Declare(
  magic_enum
  GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
  GIT_TAG        a733a2ea665ca5d72b7270f0334bf2e7b82bd0cc
)
FetchContent_MakeAvailable(magic_enum)
include_directories(${magic_enum_SOURCE_DIR}/include)

# We enabled NDEBUG even for debug builds because a gcc bug leads to a spurious assert() deep
# in the eigen3 library
add_definitions(-DNDEBUG)

set(CAFFE2_USE_CUDNN ON CACHE BOOL "Enable cuDNN support")

# Currently needed for gcc 12+
set(ENV{NVCC_APPEND_FLAGS} "-allow-unsupported-compiler")

# First-class CUDA support
project(alphazero LANGUAGES C CXX CUDA)

option(ADD_DEBUG_SYMBOLS "Add -g to release builds" OFF)
if(ADD_DEBUG_SYMBOLS)
  add_compile_options($<$<CONFIG:Release>:-g>)
endif()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Boost COMPONENTS program_options filesystem json REQUIRED)
find_package(GTest REQUIRED)
find_package(spdlog REQUIRED)
find_package(Protobuf REQUIRED)

set(ONNX_FOUND FALSE)

# 1) If someone provided ONNXConfig.cmake (custom build), use it
find_package(ONNX QUIET)
if(ONNX_FOUND)
  message(STATUS "Found ONNX via ONNXConfig.cmake")
else()
  # 2) Manual find: headers + library
  find_path(ONNX_INCLUDE_DIR onnx/onnx_pb.h
    PATHS /usr/include /usr/local/include
  )
  find_library(ONNX_LIBRARY NAMES onnx
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib
  )
  if(ONNX_INCLUDE_DIR AND ONNX_LIBRARY)
    set(ONNX_FOUND TRUE)
    message(STATUS "Found ONNX headers at ${ONNX_INCLUDE_DIR}")
    message(STATUS "Found ONNX library at ${ONNX_LIBRARY}")
  endif()
endif()

if(NOT ONNX_FOUND)
  message(FATAL_ERROR "ONNX headers/lib not found. Install libonnx-dev or build ONNX from source.")
endif()

find_package(CUDAToolkit REQUIRED)

set(TENSORRT_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
set(TENSORRT_LIB_DIR     /usr/lib/x86_64-linux-gnu)

find_library(NVINFER_LIB       NAMES nvinfer       HINTS ${TENSORRT_LIB_DIR})
find_library(NVONNXPARSER_LIB  NAMES nvonnxparser  HINTS ${TENSORRT_LIB_DIR})
find_library(NVINFERPLUGIN_LIB NAMES nvinfer_plugin HINTS ${TENSORRT_LIB_DIR})

set(TENSORRT_LIBS
  ${NVINFER_LIB}
  ${NVONNXPARSER_LIB}
  ${NVINFERPLUGIN_LIB}
)

set(CMAKE_CXX_STANDARD  23)

set(CMAKE_CXX_FLAGS_DEBUG "-O2 -g -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# If ADD_DEBUG_SYMBOLS is set, we add -g to CMAKE_CXX_FLAGS_RELEASE
if(ADD_DEBUG_SYMBOLS)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wshadow -Wno-array-bounds ${EXTRA_DEFINITIONS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-backtrace-limit=50")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")

# -fconcepts-diagnostics-depth works for gcc but not clang. Our .clangd filters it out:
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts-diagnostics-depth=5")

# We don't set this by default, because clangd doesn't recognize this option, and because we use
# clangd for code completion in Visual Studio Code.
#
# Uncomment this line to enable concepts diagnostics depth
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts-diagnostics-depth=5")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/lib)

set(TEST_COMPILE_DEFINITIONS MIT_TEST_MODE)
set(TEST_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:-fno-tree-vectorize>)

include_directories("cpp/include")
include_directories("cpp")  # for inline/ subdirectory
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${MY_EIGENRAND_DIR})
include_directories(${GTEST_INCLUDE_DIRS})

# Clear the targets file at the start of the configuration
set(_TARGETS_JSON_FIRST TRUE CACHE INTERNAL "")
file(WRITE ${CMAKE_BINARY_DIR}/targets.json "[\n")

include(cmake/GameMacros.cmake)

# Need /usr/local/cuda/lib64 in the LD_LIBRARY_PATH to find cupti
link_directories(/usr/local/cuda/lib64)
set(COMMON_EXTERNAL_LIBS
    ${Boost_LIBRARIES}
    cupti
    spdlog::spdlog
    CUDA::cudart
    ${TENSORRT_LIBS}
    onnx
    protobuf::libprotobuf
    )

set(GTEST_LIBS GTest::GTest GTest::Main gtest)

add_subdirectory("cpp/src/util")
add_subdirectory("cpp/src/core")
add_subdirectory("cpp/src/games/blokus")
add_subdirectory("cpp/src/games/chess")
add_subdirectory("cpp/src/games/connect4")
add_subdirectory("cpp/src/games/hex")
add_subdirectory("cpp/src/games/nim")
add_subdirectory("cpp/src/games/othello")
add_subdirectory("cpp/src/games/tictactoe")
add_subdirectory("cpp/src/games/stochastic_nim")
add_subdirectory("cpp/src/goldenfile_tests")

# Finalize targets.json
file(APPEND ${CMAKE_BINARY_DIR}/targets.json "\n]\n")

# https://stackoverflow.com/a/31390758/543913
macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

# print_all_variables()
