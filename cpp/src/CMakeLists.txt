# 1) the “core” util objects (everything *except* mit stuff)
add_library(util_core_objs OBJECT
  core/GameServerBase.cpp
  core/LoopControllerClient.cpp
  core/NeuralNet.cpp
  util/BoostUtil.cpp
  util/CudaUtil.cpp
  util/GTestUtil.cpp
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
add_test(
  NAME lru_cache_tests
  TEST util/main/LRUCacheTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME finite_groups_tests
  TEST util/main/FiniteGroupsTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME cpp_util_tests
  TEST util/main/CppUtilTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME meta_programming_tests
  TEST util/main/MetaProgrammingTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME bitmap_util_tests
  TEST util/main/BitMapUtilTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME gaussian1d_tests
  TEST util/main/Gaussian1DTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME recycling_alloc_pool_tests
  TEST util/main/RecyclingAllocPoolTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME random_tests
  TEST util/main/RandomTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME indexed_dispatcher_tests
  TEST util/main/IndexedDispatcherTests.cpp
  EXTRA_LIBS util_lib_test
)
