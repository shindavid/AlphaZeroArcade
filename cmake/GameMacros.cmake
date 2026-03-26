# Function to append target metadata to targets.json (one JSON object per line).
# Usage: append_target_metadata(<target> <tag1> [tag2 ...])
function(append_target_metadata target)
    set(tags ${ARGN})

    get_target_property(target_type ${target} TYPE)
    if(target_type STREQUAL "EXECUTABLE")
        get_target_property(output_dir ${target} RUNTIME_OUTPUT_DIRECTORY)
        if(NOT output_dir)
            set(output_dir "${CMAKE_BINARY_DIR}/bin")
        endif()
        get_target_property(output_name ${target} OUTPUT_NAME)
        if(NOT output_name)
            set(output_name ${target})
        endif()
    elseif(target_type STREQUAL "SHARED_LIBRARY")
        get_target_property(output_dir ${target} LIBRARY_OUTPUT_DIRECTORY)
        if(NOT output_dir)
            set(output_dir "${CMAKE_BINARY_DIR}/lib")
        endif()
        get_target_property(output_name ${target} OUTPUT_NAME)
        if(NOT output_name)
            set(output_name ${target})
        endif()
        set(output_name "lib${output_name}.so")
    else()
        return()
    endif()

    # Build tags JSON array
    set(tags_json "[")
    set(first TRUE)
    foreach(tag IN LISTS tags)
        if(first)
            string(APPEND tags_json "\"${tag}\"")
            set(first FALSE)
        else()
            string(APPEND tags_json ", \"${tag}\"")
        endif()
    endforeach()
    string(APPEND tags_json "]")

    # Add comma separator before all entries except the first
    if(_TARGETS_JSON_FIRST)
        set(_TARGETS_JSON_FIRST FALSE CACHE INTERNAL "")
    else()
        file(APPEND ${CMAKE_BINARY_DIR}/targets.json ",\n")
    endif()

    file(APPEND ${CMAKE_BINARY_DIR}/targets.json
        "{\n"
        "  \"name\": \"${target}\",\n"
        "  \"tags\": ${tags_json},\n"
        "  \"directory\": \"${output_dir}\",\n"
        "  \"filename\": \"${output_name}\"\n"
        "}")
endfunction()

# Function to add a game with common sources, main file, unit-test file and shared file
function(add_game)
    # 1) Parse args:
    #    no boolean options
    #    single‐value-required: NAME
    #    single-value-optional: EXE, TEST, FFI
    #    multi‐value-optional: SOURCES, INCLUDES
    cmake_parse_arguments(AG
        ""                     # <OPTIONS>
        "NAME;EXE;TEST;FFI"    # <ONE_VALUE>
        "SOURCES;INCLUDES;EXTRA_LIBS"     # <MULTI_VALUE>
        ${ARGN}
    )

    # 2) Sanity-check required parameters
    if(NOT AG_NAME OR (NOT AG_EXE AND NOT AG_TEST AND NOT AG_FFI))
        message(FATAL_ERROR
        "add_game() requires:\n"
        "  NAME   <game_name>\n"
        "and at least one of:\n"
        "  EXE    <exe_source>\n"
        "  TEST   <test_source>\n"
        "  FFI    <ffi_source>\n"
        "Optional:\n"
        "  SOURCES <common_source1> [common_source2…]\n"
        "  INCLUDES <inc_dir1> [inc_dir2…]\n"
        "  EXTRA_LIBS <lib1> [lib2…]\n"
        )
    endif()

    # 3) How many common sources?
    list(LENGTH AG_SOURCES _src_count)

    if(_src_count GREATER 0)
        # build the two object-libs
        add_library(${AG_NAME}_common_objs  OBJECT ${AG_SOURCES})
        set_target_properties(${AG_NAME}_common_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
        add_library(${AG_NAME}_common_objs_test OBJECT ${AG_SOURCES})
        set_target_properties(${AG_NAME}_common_objs_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_compile_definitions(${AG_NAME}_common_objs_test PRIVATE ${TEST_COMPILE_DEFINITIONS})
        target_compile_options(${AG_NAME}_common_objs_test PRIVATE ${TEST_COMPILE_OPTIONS})

        if(AG_INCLUDES)
        target_include_directories(${AG_NAME}_common_objs PUBLIC ${AG_INCLUDES})
        target_include_directories(${AG_NAME}_common_objs_test PUBLIC ${AG_INCLUDES})
        endif()

        set(_test_objs ${AG_NAME}_common_objs_test)
        set(_link_objs ${AG_NAME}_common_objs)

        target_link_libraries(${AG_NAME}_common_objs PUBLIC ${COMMON_EXTERNAL_LIBS} ${AG_EXTRA_LIBS})
        target_link_libraries(${AG_NAME}_common_objs_test PUBLIC ${COMMON_EXTERNAL_LIBS} ${AG_EXTRA_LIBS})
    else()
        # no common sources → omit object-libs
        set(_test_objs)
        set(_link_objs)
    endif()

    # 4) Test executable
    if (AG_TEST)
        add_executable(${AG_NAME}_tests ${AG_TEST})
        target_compile_definitions(${AG_NAME}_tests PRIVATE ${TEST_COMPILE_DEFINITIONS})
        target_compile_options(${AG_NAME}_tests PRIVATE ${TEST_COMPILE_OPTIONS})
        target_link_libraries(${AG_NAME}_tests
            PRIVATE
            ${_test_objs}
            core_lib_test
            ${GTEST_LIBS}
            ${COMMON_EXTERNAL_LIBS}
            ${AG_EXTRA_LIBS}
        )
        set_target_properties(${AG_NAME}_tests PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests"
        )
        append_target_metadata(${AG_NAME}_tests "${AG_NAME}" "tests")
    endif()

    # 5) Normal executable
    if (AG_EXE)
        add_executable(${AG_NAME}_exe ${AG_EXE})
        target_link_libraries(${AG_NAME}_exe
            PRIVATE
            ${_link_objs}
            core_lib
            ${COMMON_EXTERNAL_LIBS}
            ${AG_EXTRA_LIBS}
        )
        set_target_properties(${AG_NAME}_exe PROPERTIES
            OUTPUT_NAME "${AG_NAME}"
        )
        append_target_metadata(${AG_NAME}_exe "${AG_NAME}")
    endif()

    # 6) FFI shared library
    if (AG_FFI)
        add_library(${AG_NAME}_ffi SHARED ${AG_FFI})
        set_target_properties(${AG_NAME}_ffi PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_link_libraries(${AG_NAME}_ffi
            PRIVATE
            ${_link_objs}
            core_lib
            ${COMMON_EXTERNAL_LIBS}
            ${AG_EXTRA_LIBS}
        )
        set_target_properties(${AG_NAME}_ffi PROPERTIES
            OUTPUT_NAME "${AG_NAME}"
        )
        append_target_metadata(${AG_NAME}_ffi "${AG_NAME}")
    endif()
endfunction()

# Function add a standalone test, separate from a game-specific test added via add_game()
function(add_test)
  # 1) Parse named args: NAME, TEST are single-value; EXTRA_LIBS is optional multi-value
  #    NO_MIT is a boolean flag that disables MIT_TEST_MODE (for tests using real sockets, etc.)
  cmake_parse_arguments(AT
    "NO_MIT"              # boolean flags
    "NAME;TEST"           # single-value keywords
    "EXTRA_LIBS"          # multi-value keyword
    ${ARGN}
  )

  # 2) Sanity‐check
  if(NOT AT_NAME OR NOT AT_TEST)
    message(FATAL_ERROR
      "add_test() requires:\n"
      "  NAME      <test_target_name>\n"
      "  TEST      <test_source_file>\n"
      "Optional:\n"
      "  NO_MIT    (skip MIT_TEST_MODE)\n"
      "  EXTRA_LIBS <lib1> [lib2…]\n"
    )
  endif()

  # 3) Verify every EXTRA_LIB ends with "_test" (unless NO_MIT is set)
  if(NOT AT_NO_MIT)
    foreach(_lib IN LISTS AT_EXTRA_LIBS)
      string(LENGTH "${_lib}" _len)
      if(_len LESS 5)
        message(FATAL_ERROR "add_test(${AT_NAME}): EXTRA_LIB \"${_lib}\" must end with \"_test\".")
      endif()
      math(EXPR _start "${_len} - 5")
      string(SUBSTRING "${_lib}" ${_start} 5 _suffix)
      if(NOT _suffix STREQUAL "_test")
        message(FATAL_ERROR "add_test(${AT_NAME}): EXTRA_LIB \"${_lib}\" must end with \"_test\".")
      endif()
    endforeach()
  endif()

  # 4) Create the test executable
  add_executable(${AT_NAME} ${AT_TEST})

  # 5) Compile‐time definitions (skip MIT_TEST_MODE if NO_MIT is set)
  if(NOT AT_NO_MIT)
    target_compile_definitions(${AT_NAME}
      PRIVATE ${TEST_COMPILE_DEFINITIONS}
    )

    # 5a) Disable vectorization for standalone tests (C++ only)
    target_compile_options(${AT_NAME} PRIVATE ${TEST_COMPILE_OPTIONS})
  endif()

  # 6) Link libraries: EXTRA_LIBS → util_lib_test, core_lib_test, etc.
  #    plus COMMON_EXTERNAL_LIBS and GTEST_LIBS
  target_link_libraries(${AT_NAME}
    PRIVATE
      ${AT_EXTRA_LIBS}
      ${COMMON_EXTERNAL_LIBS}
      ${GTEST_LIBS}
  )

  # 7) Put the binary into the tests/ folder
  set_target_properties(${AT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/tests"
  )

  # 8) Metadata for listing
  append_target_metadata(${AT_NAME} "aux" "tests")
endfunction()
