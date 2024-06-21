add_library(othello-lib
        Game.cpp
        flip_sse.cpp
        )

add_executable(othello
        main/Othello.cpp
        )

add_executable(othello-tests
        main/UnitTests.cpp
        )

set(LOCAL_LIBS othello-lib)
set(ALL_LIBS ${ALL_LIBS} ${LOCAL_LIBS} PARENT_SCOPE)
set(LOCAL_EXECUTABLES othello othello-tests)
set(ALL_EXECUTABLES ${ALL_EXECUTABLES} ${LOCAL_EXECUTABLES} PARENT_SCOPE)

foreach(lib ${LOCAL_LIBS})
    foreach(tgt ${LOCAL_EXECUTABLES})
        target_link_libraries(${tgt} othello-lib)
    endforeach()
endforeach()
