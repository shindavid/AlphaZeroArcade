set(COMMON_SOURCES
    Game.cpp
    Pieces.cpp
    Types.cpp
)

add_executable(blokus_tests main/UnitTests.cpp ${COMMON_SOURCES})
add_executable(blokus_exe main/Blokus.cpp ${COMMON_SOURCES})
add_library(blokus_ffi SHARED shared/Blokus_ffi.cpp ${COMMON_SOURCES})
set_target_properties(blokus_ffi PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(LINK_LIBS util_lib ${TORCH_LIBRARIES} ${Boost_LIBRARIES})
set(TARGETS blokus_tests blokus_exe blokus_ffi)
foreach(target ${TARGETS})
    target_link_libraries(${target} PRIVATE ${LINK_LIBS})
endforeach()

set_target_properties(blokus_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests")
set_target_properties(blokus_exe PROPERTIES OUTPUT_NAME "blokus")
set_target_properties(blokus_ffi PROPERTIES OUTPUT_NAME "blokus")

append_target_metadata("blokus" ${TARGETS})
