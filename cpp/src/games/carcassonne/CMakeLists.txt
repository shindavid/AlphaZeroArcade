set(COMMON_SOURCES
    LookupTables.cpp
)

add_library(carcassonne_common_objs OBJECT ${COMMON_SOURCES})
set_target_properties(carcassonne_common_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(carcassonne_tests main/UnitTests.cpp)
add_executable(carcassonne_exe main/Carcassonne.cpp)
add_library(carcassonne_ffi SHARED shared/Carcassonne_ffi.cpp)
set_target_properties(carcassonne_ffi PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(carcassonne_tests PRIVATE ${GTEST_LIBS})

set(LINK_LIBS carcassonne_common_objs util_lib ${COMMON_EXTERNAL_LIBS})
set(TARGETS carcassonne_tests carcassonne_exe carcassonne_ffi)
foreach(target ${TARGETS})
    target_link_libraries(${target} PRIVATE ${LINK_LIBS})
endforeach()

set_target_properties(carcassonne_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")
set_target_properties(carcassonne_exe PROPERTIES OUTPUT_NAME "carcassonne")
set_target_properties(carcassonne_ffi PROPERTIES OUTPUT_NAME "carcassonne")

append_target_metadata("carcassonne" ${TARGETS})
