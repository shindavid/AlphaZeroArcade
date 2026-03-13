add_library(util_objs OBJECT
  BoostUtil.cpp
  CudaUtil.cpp
  GTestUtil.cpp
  SocketUtil.cpp
)
set_target_properties(util_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(util_objs PRIVATE ${COMMON_EXTERNAL_LIBS})

# non-test lib (util only)
add_library(util_lib STATIC $<TARGET_OBJECTS:util_objs>)
target_link_libraries(util_lib PRIVATE ${COMMON_EXTERNAL_LIBS})

# test lib (util only)
add_library(util_lib_test STATIC $<TARGET_OBJECTS:util_objs>)
target_compile_definitions(util_lib_test PRIVATE ${TEST_COMPILE_DEFINITIONS})
target_link_libraries(util_lib_test PRIVATE ${COMMON_EXTERNAL_LIBS})

# test executables
add_test(
  NAME eigen_util_tests
  TEST main/EigenUtilTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME math_tests
  TEST main/MathTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME compact_bit_set_tests
  TEST main/CompactBitSetTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME string_util_tests
  TEST main/StringUtilTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME boost_util_tests
  TEST main/BoostUtilTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME alloc_pool_tests
  TEST main/AllocPoolTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME cuda_util_tests
  TEST main/CudaUtilTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME tensorrt_util_tests
  TEST main/TensorRtUtilTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME exponential_decay_tests
  TEST main/ExponentialDecayTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME static_circular_buffer_tests
  TEST main/StaticCircularBufferTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME mit_tests
  TEST main/mit/UnitTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME lru_cache_tests
  TEST main/LRUCacheTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME finite_groups_tests
  TEST main/FiniteGroupsTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME cpp_util_tests
  TEST main/CppUtilTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME meta_programming_tests
  TEST main/MetaProgrammingTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME bitmap_util_tests
  TEST main/BitMapUtilTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME gaussian1d_tests
  TEST main/Gaussian1DTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME recycling_alloc_pool_tests
  TEST main/RecyclingAllocPoolTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME random_tests
  TEST main/RandomTests.cpp
  EXTRA_LIBS util_lib_test
)
add_test(
  NAME indexed_dispatcher_tests
  TEST main/IndexedDispatcherTests.cpp
  EXTRA_LIBS util_lib_test
)
