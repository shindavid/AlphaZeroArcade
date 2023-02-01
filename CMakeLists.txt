cmake_minimum_required(VERSION 3.17)

set(CMAKE_DEBUG_POSTFIX d)
project(alphazero)

if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DNDEBUG)
endif()

find_package (Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Torch REQUIRED PATHS ${MY_TORCH_DIR})
find_package(Boost COMPONENTS program_options filesystem  REQUIRED)

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-array-bounds -O3 ${TORCH_CXX_FLAGS} -fconcepts-diagnostics-depth=5 ${EXTRA_DEFINITIONS}")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/lib)

include_directories("cpp")
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${MY_EIGENRAND_DIR})
include_directories(${MY_TINYEXPR_DIR})

add_executable(c4_competitive_self_play cpp/connect4/main/CompetitiveSelfPlay.cpp)
add_executable(c4_dshin_scratch cpp/connect4/main/DshinScratch.cpp)
add_executable(c4_generate_oracle_labeled_games cpp/connect4/main/GenerateOracleLabeledGames.cpp)
add_executable(c4_mcts_vs_perfect cpp/connect4/main/MctsVsPerfect.cpp)
add_executable(c4_play_vs_cpu cpp/connect4/main/PlayVsCpu.cpp)
add_executable(c4_training_self_play cpp/connect4/main/TrainingSelfPlay.cpp)

set(executables
        c4_competitive_self_play
        c4_dshin_scratch
        c4_generate_oracle_labeled_games
        c4_mcts_vs_perfect
        c4_play_vs_cpu
        c4_training_self_play
        )

add_subdirectory("cpp/util")
add_subdirectory("cpp/connect4")

set(libraries util connect4)

foreach(tgt ${executables} ${libraries})
    target_link_libraries(${tgt} ${TORCH_LIBRARIES})
    target_link_libraries(${tgt} ${Boost_LIBRARIES})
endforeach()

link_directories(${Boost_LIBRARY_DIRS})

foreach(tgt ${executables})
    foreach(lib ${libraries})
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

#print_all_variables()
