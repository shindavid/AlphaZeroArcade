add_library(util_lib
        core/LoopControllerClient.cpp
        util/BoostUtil.cpp
        util/CudaUtil.cpp
        util/KeyValueDumper.cpp
        util/LoggingUtil.cpp
        util/Math.cpp
        util/RepoUtil.cpp
        util/SocketUtil.cpp
        util/StaticVars.cpp
        )

set_target_properties(util_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(TARGETS util_tests)

add_executable(util_tests util/main/UnitTests.cpp)
set_target_properties(util_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")

target_link_libraries(util_lib PRIVATE ${COMMON_EXTERNAL_LIBS})
target_link_libraries(util_tests PRIVATE util_lib ${GTEST_LIBS} ${COMMON_EXTERNAL_LIBS})

append_target_metadata("aux" ${TARGETS})
