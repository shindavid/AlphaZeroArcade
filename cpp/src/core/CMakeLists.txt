add_library(core_objs OBJECT
  GameServerBase.cpp
  NeuralNet.cpp
)
set_target_properties(core_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(core_objs PRIVATE ${COMMON_EXTERNAL_LIBS})

# non-test lib (core + util)
add_library(core_lib STATIC $<TARGET_OBJECTS:core_objs> $<TARGET_OBJECTS:util_objs>)
target_link_libraries(core_lib PRIVATE ${COMMON_EXTERNAL_LIBS})

# test lib (core + util)
add_library(core_lib_test STATIC $<TARGET_OBJECTS:core_objs> $<TARGET_OBJECTS:util_objs>)
target_compile_definitions(core_lib_test PRIVATE ${TEST_COMPILE_DEFINITIONS})
target_link_libraries(core_lib_test PRIVATE ${COMMON_EXTERNAL_LIBS})

# test executables
add_test(
  NAME loop_controller_client_tests
  TEST main/LoopControllerClientTests.cpp
  EXTRA_LIBS core_lib_test
)
