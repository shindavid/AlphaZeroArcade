cmake_minimum_required(VERSION 3.17)

set(CMAKE_DEBUG_POSTFIX d)
project(alphazero)


if(CONDA_PREFIX)
    # Append to CMAKE_PREFIX_PATH and link_directories
    list(APPEND CMAKE_PREFIX_PATH "${CONDA_PREFIX}")
    link_directories(${CONDA_PREFIX}/lib)
endif()

# We enabled NDEBUG even for debug builds because a gcc bug leads to a spurious assert() deep
# in the eigen3 library
add_definitions(-DNDEBUG)

find_package (Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Torch REQUIRED PATHS ${MY_TORCH_DIR})
find_package(Boost COMPONENTS program_options filesystem json REQUIRED)

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-array-bounds -O3 ${TORCH_CXX_FLAGS} -fconcepts-diagnostics-depth=5 -ftemplate-backtrace-limit=50 ${EXTRA_DEFINITIONS}")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/lib)

SET(EXTERNAL_LIBS ${TORCH_LIBRARIES} ${Boost_LIBRARIES})
set(COMMON_LIBS "")  # each executable of ALL_EXECUTABLES will be linked with these
set(ALL_LIBS "")  # these are linked with each lib of EXTERNAL_LIBS
set(ALL_EXECUTABLES "")

include_directories("cpp/include")
include_directories("cpp")  # for inline/ subdirectory
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${MY_EIGENRAND_DIR})
include_directories(${MY_TINYEXPR_DIR})

add_subdirectory("cpp/src")
add_subdirectory("games/connect4/cpp/src")
add_subdirectory("games/othello/cpp/src")
add_subdirectory("games/tictactoe/cpp/src")

foreach(tgt ${ALL_LIBS})
    foreach(lib ${EXTERNAL_LIBS})
        target_link_libraries(${tgt} ${lib})
    endforeach()
endforeach()

foreach(tgt ${ALL_EXECUTABLES})
    foreach(lib ${COMMON_LIBS})
        target_link_libraries(${tgt} ${lib})
    endforeach()
    set_target_properties(${tgt} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    set_property(TARGET ${tgt} PROPERTY CXX_STANDARD 20)
endforeach()

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
