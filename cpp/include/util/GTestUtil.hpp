#pragma once

#include <gtest/gtest.h>

/*
 * This contains utility macros for Google Test.
 *
 * If you want to give a Google test class access to private members of another class, you can use
 * the FRIEND_GTEST macro. This must be coupled with a forward declaration of the test class
 * outside of the namespace of the class being tested.
 *
 * Usage:
 *
 * GTEST_FORWARD_DECLARE(TestClassName);
 * GTEST_FORWARD_DECLARE(TestClassName, test_name1);
 * GTEST_FORWARD_DECLARE(TestClassName, test_name2);
 *
 * namespace n {
 * class ClassBeingTested {
 *   ...
 *   FRIEND_GTEST(TestClassName);
 *   FRIEND_GTEST(TestClassName, test_name1);
 *   FRIEND_GTEST(TestClassName, test_name2);
 * };
 * }  // namespace n
 *
 * class TestClassName : public testing::Test {
 *   // the FRIEND_GTEST(TestClassName) call allows us to use private members of ClassBeingTested
 *   // within this class
 *   ...
 * };
 *
 * TEST_F(TestClassName, test_name1) {
 *   // the FRIEND_GTEST(TestClassName, test_name1) call allows us to use private members of
 *   // ClassBeingTested within this test
 *   ...
 * }
 *
 * TEST_F(TestClassName, test_name2) {
 *   // the FRIEND_GTEST(TestClassName, test_name2) call allows us to use private members of
 *   // ClassBeingTested within this test
 *   ...
 * }
 */

// Forward declaration with either one or two arguments
#define GTEST_FORWARD_DECLARE_HELPER(test_case_name, test_name) \
  class GTEST_TEST_CLASS_NAME_(test_case_name, test_name);

#define GTEST_FORWARD_DECLARE_1(test_case_name) class test_case_name;

#define GTEST_FORWARD_DECLARE_2(test_case_name, test_name) \
  GTEST_FORWARD_DECLARE_HELPER(test_case_name, test_name)

// Detect if one or two arguments are provided to GTEST_FORWARD_DECLARE
#define GTEST_FORWARD_DECLARE(...)                                                               \
  GTEST_FORWARD_DECLARE_GET_MACRO(__VA_ARGS__, GTEST_FORWARD_DECLARE_2, GTEST_FORWARD_DECLARE_1) \
  (__VA_ARGS__)

// Friend declaration with either one or two arguments
#define FRIEND_GTEST_HELPER(test_case_name, test_name) \
  friend class ::GTEST_TEST_CLASS_NAME_(test_case_name, test_name);

#define FRIEND_GTEST_1(test_case_name) friend class ::test_case_name;

#define FRIEND_GTEST_2(test_case_name, test_name) FRIEND_GTEST_HELPER(test_case_name, test_name)

// Detect if one or two arguments are provided to FRIEND_GTEST
#define FRIEND_GTEST(...) \
  FRIEND_GTEST_GET_MACRO(__VA_ARGS__, FRIEND_GTEST_2, FRIEND_GTEST_1)(__VA_ARGS__)

// Helper macro to detect how many arguments are provided
#define GTEST_FORWARD_DECLARE_GET_MACRO(_1, _2, NAME, ...) NAME
#define FRIEND_GTEST_GET_MACRO(_1, _2, NAME, ...) NAME

// Dispatches to standard gtest main function, while adding LoggingUtil cmdline params
int launch_gtest(int argc, char** argv);
