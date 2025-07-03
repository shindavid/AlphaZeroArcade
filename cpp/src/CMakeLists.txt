# 1) object library containing *all* the .cpp files
add_library(util_objs OBJECT
  core/GameServerBase.cpp
  core/LoopControllerClient.cpp
  util/BoostUtil.cpp
  util/CudaUtil.cpp
  util/GTestUtil.cpp
  util/KeyValueDumper.cpp
  util/LoggingUtil.cpp
  util/Math.cpp
  util/RepoUtil.cpp
  util/SocketUtil.cpp
  util/StaticVars.cpp
)
set_target_properties(util_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(util_objs PRIVATE ${COMMON_EXTERNAL_LIBS})

add_library(util_lib STATIC $<TARGET_OBJECTS:util_objs>)
target_link_libraries(util_lib PRIVATE ${COMMON_EXTERNAL_LIBS})

add_library(util_lib_test STATIC $<TARGET_OBJECTS:util_objs>)
target_compile_definitions(util_lib_test PRIVATE ${TEST_COMPILE_DEFINITIONS})
target_link_libraries(util_lib_test PRIVATE ${COMMON_EXTERNAL_LIBS})

# 4) your test executable
add_executable(util_tests util/main/UnitTests.cpp)
set_target_properties(util_tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests"
)
target_compile_definitions(util_tests PRIVATE ${TEST_COMPILE_DEFINITIONS})
target_link_libraries(util_tests PRIVATE util_lib_test ${GTEST_LIBS} ${COMMON_EXTERNAL_LIBS})

append_target_metadata("aux" util_tests)

# pull in tools/*
add_subdirectory(tools)
