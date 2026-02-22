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
project(alphazero LANGUAGES CXX CUDA)

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

# Function to add a game with common sources, main file, unit-test file and shared file
function(add_game)
    # 1) Parse args:
    #    no boolean options
    #    single‐value-required: NAME
    #    single-value-optional: EXE, TEST, FFI
    #    multi‐value-optional: SOURCES, INCLUDES
    cmake_parse_arguments(AG
        ""                     # <OPTIONS>
        "NAME;EXE;TEST;FFI"    # <ONE_VALUE>
        "SOURCES;INCLUDES"     # <MULTI_VALUE>
        ${ARGN}
    )

    # 2) Sanity-check required parameters
    if(NOT AG_NAME OR (NOT AG_EXE AND NOT AG_TEST AND NOT AG_FFI))
        message(FATAL_ERROR
        "add_game() requires:\n"
        "  NAME   <game_name>\n"
        "and at least one of:\n"
        "  EXE    <exe_source>\n"
        "  TEST   <test_source>\n"
        "  FFI    <ffi_source>\n"
        "Optional:\n"
        "  SOURCES <common_source1> [common_source2…]\n"
        "  INCLUDES <inc_dir1> [inc_dir2…]\n"
        )
    endif()

    # 3) How many common sources?
    list(LENGTH AG_SOURCES _src_count)

    if(_src_count GREATER 0)
        # build the two object-libs
        add_library(${AG_NAME}_common_objs  OBJECT ${AG_SOURCES})
        set_target_properties(${AG_NAME}_common_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
        add_library(${AG_NAME}_common_objs_test OBJECT ${AG_SOURCES})
        set_target_properties(${AG_NAME}_common_objs_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_compile_definitions(${AG_NAME}_common_objs_test PRIVATE ${TEST_COMPILE_DEFINITIONS})
        target_compile_options(${AG_NAME}_common_objs_test PRIVATE ${TEST_COMPILE_OPTIONS})

        if(AG_INCLUDES)
        target_include_directories(${AG_NAME}_common_objs PUBLIC ${AG_INCLUDES})
        target_include_directories(${AG_NAME}_common_objs_test PUBLIC ${AG_INCLUDES})
        endif()

        set(_test_objs ${AG_NAME}_common_objs_test)
        set(_link_objs ${AG_NAME}_common_objs)

        target_link_libraries(${AG_NAME}_common_objs PUBLIC ${COMMON_EXTERNAL_LIBS})
        target_link_libraries(${AG_NAME}_common_objs_test PUBLIC ${COMMON_EXTERNAL_LIBS})
    else()
        # no common sources → omit object-libs
        set(_test_objs)
        set(_link_objs)
    endif()

    # 4) Test executable
    if (AG_TEST)
        add_executable(${AG_NAME}_tests ${AG_TEST})
        target_compile_definitions(${AG_NAME}_tests PRIVATE ${TEST_COMPILE_DEFINITIONS})
        target_compile_options(${AG_NAME}_tests PRIVATE ${TEST_COMPILE_OPTIONS})
        target_link_libraries(${AG_NAME}_tests
            PRIVATE
            ${_test_objs}
            util_lib_test
            ${GTEST_LIBS}
            ${COMMON_EXTERNAL_LIBS}
        )
        set_target_properties(${AG_NAME}_tests PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests"
        )
        append_target_metadata("${AG_NAME}" ${AG_NAME}_tests)
    endif()

    # 5) Normal executable
    if (AG_EXE)
        add_executable(${AG_NAME}_exe ${AG_EXE})
        target_link_libraries(${AG_NAME}_exe
            PRIVATE
            ${_link_objs}
            util_lib
            ${COMMON_EXTERNAL_LIBS}
        )
        set_target_properties(${AG_NAME}_exe PROPERTIES
            OUTPUT_NAME "${AG_NAME}"
        )
        append_target_metadata("${AG_NAME}" ${AG_NAME}_exe)
    endif()

    # 6) FFI shared library
    if (AG_FFI)
        add_library(${AG_NAME}_ffi SHARED ${AG_FFI})
        set_target_properties(${AG_NAME}_ffi PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_link_libraries(${AG_NAME}_ffi
            PRIVATE
            ${_link_objs}
            util_lib
            ${COMMON_EXTERNAL_LIBS}
        )
        set_target_properties(${AG_NAME}_ffi PROPERTIES
            OUTPUT_NAME "${AG_NAME}"
        )
        append_target_metadata("${AG_NAME}" ${AG_NAME}_ffi)
    endif()
endfunction()

# Function add a standalone test, separate from a game-specific test added via add_game()
function(add_test)
  # 1) Parse named args: NAME, TEST are single-value; EXTRA_LIBS is optional multi-value
  cmake_parse_arguments(AT
    ""                    # no boolean flags
    "NAME;TEST"           # single-value keywords
    "EXTRA_LIBS"          # multi-value keyword
    ${ARGN}
  )

  # 2) Sanity‐check
  if(NOT AT_NAME OR NOT AT_TEST)
    message(FATAL_ERROR
      "add_test() requires:\n"
      "  NAME      <test_target_name>\n"
      "  TEST      <test_source_file>\n"
      "Optional:\n"
      "  EXTRA_LIBS <lib1> [lib2…]\n"
    )
  endif()

  # 3) Verify every EXTRA_LIB ends with "_test"
  foreach(_lib IN LISTS AT_EXTRA_LIBS)
    string(LENGTH "${_lib}" _len)
    if(_len LESS 5)
      message(FATAL_ERROR "add_test(${AT_NAME}): EXTRA_LIB \"${_lib}\" must end with \"_test\".")
    endif()
    math(EXPR _start "${_len} - 5")
    string(SUBSTRING "${_lib}" ${_start} 5 _suffix)
    if(NOT _suffix STREQUAL "_test")
      message(FATAL_ERROR "add_test(${AT_NAME}): EXTRA_LIB \"${_lib}\" must end with \"_test\".")
    endif()
  endforeach()

  # 4) Create the test executable
  add_executable(${AT_NAME} ${AT_TEST})

  # 5) Compile‐time definitions
  target_compile_definitions(${AT_NAME}
    PRIVATE ${TEST_COMPILE_DEFINITIONS}
  )

  # 5a) Disable vectorization for standalone tests (C++ only)
  target_compile_options(${AT_NAME} PRIVATE ${TEST_COMPILE_OPTIONS})

  # 6) Link libraries: EXTRA_LIBS → util_lib_test, tictactoe_common_objs_test, etc.
  #    plus COMMON_EXTERNAL_LIBS and GTEST_LIBS
  target_link_libraries(${AT_NAME}
    PRIVATE
      ${AT_EXTRA_LIBS}
      ${COMMON_EXTERNAL_LIBS}
      ${GTEST_LIBS}
  )

  # 7) Put the binary into the tests/ folder
  set_target_properties(${AT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests"
  )

  # 8) Metadata for listing
  append_target_metadata("aux" ${AT_NAME})
endfunction()

add_subdirectory("cpp/src")
add_subdirectory("cpp/src/games/blokus")
#add_subdirectory("cpp/src/games/chess")
add_subdirectory("cpp/src/games/connect4")
add_subdirectory("cpp/src/games/hex")
add_subdirectory("cpp/src/games/nim")
add_subdirectory("cpp/src/games/othello")
add_subdirectory("cpp/src/games/tictactoe")
add_subdirectory("cpp/src/games/stochastic_nim")
add_subdirectory("cpp/src/alpha0")
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
