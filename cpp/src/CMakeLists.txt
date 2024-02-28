add_library(alphazero-lib
        core/TrainingServerClient.cpp
        util/BoostUtil.cpp
        util/CudaUtil.cpp
        util/KeyValueDumper.cpp
        util/LoggingUtil.cpp
        util/Math.cpp
        util/RepoUtil.cpp
        util/SocketUtil.cpp
        util/StaticVars.cpp
        ${MY_TINYEXPR_DIR}/tinyexpr.c
        )

set(LOCAL_LIBS alphazero-lib)
set(COMMON_LIBS ${COMMON_LIBS} ${LOCAL_LIBS} PARENT_SCOPE)
set(ALL_LIBS ${ALL_LIBS} ${LOCAL_LIBS} PARENT_SCOPE)

add_executable(alphazero-lib-tests
        util/main/UnitTests.cpp
        )

set(LOCAL_EXECUTABLES alphazero-lib-tests)
set(ALL_EXECUTABLES ${ALL_EXECUTABLES} ${LOCAL_EXECUTABLES} PARENT_SCOPE)
