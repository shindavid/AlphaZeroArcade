# 1) the “core” util objects (everything *except* mit stuff)
add_library(util_core_objs OBJECT
  core/GameServerBase.cpp
  core/LoopControllerClient.cpp
  util/BoostUtil.cpp
  util/CudaUtil.cpp
  util/GTestUtil.cpp
  util/LoggingUtil.cpp
  util/Math.cpp
  util/SocketUtil.cpp
)
set_target_properties(util_core_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(util_core_objs PRIVATE ${COMMON_EXTERNAL_LIBS})

# 3) non-test lib
add_library(util_lib STATIC $<TARGET_OBJECTS:util_core_objs>)
target_link_libraries(util_lib PRIVATE ${COMMON_EXTERNAL_LIBS})

# 4) test lib
add_library(util_lib_test STATIC $<TARGET_OBJECTS:util_core_objs>)
target_compile_definitions(util_lib_test PRIVATE ${TEST_COMPILE_DEFINITIONS})
target_link_libraries(util_lib_test PRIVATE ${COMMON_EXTERNAL_LIBS})

# 5) test executables
add_test(
  NAME util_tests
  TEST util/main/UnitTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME mit_tests
  TEST util/main/mit/UnitTests.cpp
  EXTRA_LIBS util_lib_test
)

# 6) pull in tools/*
add_subdirectory(tools)
