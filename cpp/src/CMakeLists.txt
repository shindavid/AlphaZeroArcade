# 1) the “core” util objects (everything *except* mit stuff)
add_library(util_core_objs OBJECT
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
set_target_properties(util_core_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(util_core_objs PRIVATE ${COMMON_EXTERNAL_LIBS})

# 2) the mit-specific object
add_library(util_mit_objs OBJECT
  util/mit/scheduler.cpp
)
set_target_properties(util_mit_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(util_mit_objs PRIVATE ${TEST_COMPILE_DEFINITIONS})
target_link_libraries(util_mit_objs PRIVATE ${COMMON_EXTERNAL_LIBS})

# 3) non-test lib
add_library(util_lib STATIC $<TARGET_OBJECTS:util_core_objs>)
target_link_libraries(util_lib PRIVATE ${COMMON_EXTERNAL_LIBS})

# 4) test lib
add_library(util_lib_test STATIC $<TARGET_OBJECTS:util_core_objs> $<TARGET_OBJECTS:util_mit_objs>)
target_compile_definitions(util_lib_test PRIVATE ${TEST_COMPILE_DEFINITIONS})
target_link_libraries(util_lib_test PRIVATE ${COMMON_EXTERNAL_LIBS})

# 5) test executable
add_test(
  NAME util_tests
  TEST util/main/UnitTests.cpp
  EXTRA_LIBS util_lib_test
)

# 6) pull in tools/*
add_subdirectory(tools)
