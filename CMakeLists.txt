cmake_minimum_required(VERSION 3.17)

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
project(alphazero LANGUAGES CXX CUDA)

find_package (Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Boost COMPONENTS program_options filesystem json REQUIRED)
find_package(GTest REQUIRED)
find_package(spdlog REQUIRED)

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

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# If ADD_DEBUG_SYMBOLS is set, we add -g to CMAKE_CXX_FLAGS_RELEASE
if(ADD_DEBUG_SYMBOLS)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wshadow -Wno-array-bounds ${EXTRA_DEFINITIONS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-backtrace-limit=50")

# We don't set this by default, because clangd doesn't recognize this option, and because we use
# clangd for code completion in Visual Studio Code.
#
# Uncomment this line to enable concepts diagnostics depth
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts-diagnostics-depth=5")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/lib)

include_directories("cpp/include")
include_directories("cpp")  # for inline/ subdirectory
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${MY_EIGENRAND_DIR})
include_directories(${GTEST_INCLUDE_DIRS})

# Clear the targets file at the start of the configuration
file(WRITE ${CMAKE_BINARY_DIR}/targets.txt "")

# Function to append target metadata to the targets.txt file
function(append_target_metadata game_name)
    foreach(target ${ARGN})
        get_target_property(target_type ${target} TYPE)
        if(target_type STREQUAL "EXECUTABLE")
            get_target_property(output_dir ${target} RUNTIME_OUTPUT_DIRECTORY)
            if(NOT output_dir)
                set(output_dir "${CMAKE_BINARY_DIR}/bin")
            endif()
            get_target_property(output_name ${target} OUTPUT_NAME)
            if(NOT output_name)
                set(output_name ${target})
            endif()
            file(APPEND ${CMAKE_BINARY_DIR}/targets.txt "${game_name} ${target} ${output_dir} ${output_name}\n")
        elseif(target_type STREQUAL "SHARED_LIBRARY")
            get_target_property(output_dir ${target} LIBRARY_OUTPUT_DIRECTORY)
            if(NOT output_dir)
                set(output_dir "${CMAKE_BINARY_DIR}/lib")
            endif()
            get_target_property(output_name ${target} OUTPUT_NAME)
            if(NOT output_name)
                set(output_name ${target})
            endif()
            file(APPEND ${CMAKE_BINARY_DIR}/targets.txt "${game_name} ${target} ${output_dir} lib${output_name}.so\n")
        endif()
    endforeach()
endfunction()

# Need /usr/local/cuda/lib64 in the LD_LIBRARY_PATH to find cupti
link_directories(/usr/local/cuda/lib64)
set(COMMON_EXTERNAL_LIBS ${Boost_LIBRARIES} cupti spdlog::spdlog CUDA::cudart ${TENSORRT_LIBS})
set(GTEST_LIBS GTest::GTest GTest::Main gtest)

add_subdirectory("cpp/src")
add_subdirectory("cpp/src/games/blokus")
add_subdirectory("cpp/src/games/chess")
add_subdirectory("cpp/src/games/connect4")
add_subdirectory("cpp/src/games/othello")
add_subdirectory("cpp/src/games/tictactoe")
add_subdirectory("cpp/src/games/nim")
add_subdirectory("cpp/src/games/stochastic_nim")
add_subdirectory("cpp/src/mcts")
add_subdirectory("cpp/src/generic_players")

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
