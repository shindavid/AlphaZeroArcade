#include "util/GTestUtil.hpp"
#include "util/StringUtil.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

TEST(StringUtil, atof_safe) {
  EXPECT_EQ(util::atof_safe("0.0"), 0.0f);
  EXPECT_EQ(util::atof_safe("5"), 5.0f);
  EXPECT_EQ(util::atof_safe("-5"), -5.0f);
  EXPECT_EQ(util::atof_safe("1.0"), 1.0f);
  EXPECT_EQ(util::atof_safe("3.14"), 3.14f);
  EXPECT_EQ(util::atof_safe("1e-3"), 0.001f);
  EXPECT_EQ(util::atof_safe("1e3"), 1000.0f);
  EXPECT_EQ(util::atof_safe("1.0e3"), 1000.0f);
  EXPECT_EQ(util::atof_safe("1.0e-3"), 0.001f);
  EXPECT_EQ(util::atof_safe("1.0e+3"), 1000.0f);
  EXPECT_EQ(util::atof_safe("-1.0e+3"), -1000.0f);

  EXPECT_THROW(util::atof_safe(""), std::invalid_argument);
  EXPECT_THROW(util::atof_safe("abc"), std::invalid_argument);
  EXPECT_THROW(util::atof_safe("1.0abc"), std::invalid_argument);
  EXPECT_THROW(util::atof_safe("1.0eabc"), std::invalid_argument);
}

TEST(StringUtil, split) {
  std::vector<std::string> result1 = util::split("a,b,c", ",");
  std::vector<std::string> result2 = util::split(" a \tb   c ");

  EXPECT_EQ(result1.size(), 3);
  EXPECT_EQ(result1[0], "a");
  EXPECT_EQ(result1[1], "b");
  EXPECT_EQ(result1[2], "c");

  EXPECT_EQ(result2.size(), 3);
  EXPECT_EQ(result2[0], "a");
  EXPECT_EQ(result2[1], "b");
  EXPECT_EQ(result2[2], "c");

  std::vector<std::string> result3;
  int n3;
  n3 = util::split(result3, "a,bb,ccc", ",");

  EXPECT_EQ(n3, 3);
  EXPECT_EQ(result3.size(), 3);
  EXPECT_EQ(result3[0], "a");
  EXPECT_EQ(result3[1], "bb");
  EXPECT_EQ(result3[2], "ccc");

  n3 = util::split(result3, "\t\taa  b\n");

  EXPECT_EQ(n3, 2);
  EXPECT_EQ(result3.size(), 3);
  EXPECT_EQ(result3[0], "aa");
  EXPECT_EQ(result3[1], "b");
  EXPECT_EQ(result3[2], "ccc");

  n3 = util::split(result3, "\t\taa  b c    d  e\n");

  EXPECT_EQ(n3, 5);
  EXPECT_EQ(result3.size(), 5);
  EXPECT_EQ(result3[0], "aa");
  EXPECT_EQ(result3[1], "b");
  EXPECT_EQ(result3[2], "c");
  EXPECT_EQ(result3[3], "d");
  EXPECT_EQ(result3[4], "e");
}

TEST(StringUtil, splitlines) {
  std::vector<std::string> result1 = util::splitlines("a\nb\nc");
  std::vector<std::string> result2 = util::splitlines("\n ac\n");
  std::vector<std::string> result3 = util::splitlines("");
  std::vector<std::string> result4 = util::splitlines("\n");

  EXPECT_EQ(result1.size(), 3);
  EXPECT_EQ(result1[0], "a");
  EXPECT_EQ(result1[1], "b");
  EXPECT_EQ(result1[2], "c");

  EXPECT_EQ(result2.size(), 2);
  EXPECT_EQ(result2[0], "");
  EXPECT_EQ(result2[1], " ac");

  EXPECT_EQ(result3.size(), 0);

  EXPECT_EQ(result4.size(), 1);
  EXPECT_EQ(result4[0], "");
}

TEST(StringUtil, ends_with) {
  EXPECT_TRUE(util::ends_with("hello", "lo"));
  EXPECT_TRUE(util::ends_with("hello", "hello"));
  EXPECT_FALSE(util::ends_with("hello", "hello world"));
  EXPECT_FALSE(util::ends_with("hello", "hello hello"));
  EXPECT_TRUE(util::ends_with("hello", ""));
  EXPECT_TRUE(util::ends_with("", ""));
  EXPECT_FALSE(util::ends_with("", "a"));
}

TEST(StringUtil, terminal_width) {
  EXPECT_EQ(util::terminal_width(""), 0);
  EXPECT_EQ(util::terminal_width("hello"), 5);
  EXPECT_EQ(util::terminal_width("\033[31m\033[00m"), 0);
  EXPECT_EQ(util::terminal_width("\033[31mhello\033[00m"), 5);
}

TEST(StringUtil, grammatically_join) {
  EXPECT_EQ(util::grammatically_join({"a"}, "and"), "a");
  EXPECT_EQ(util::grammatically_join({"a", "b"}, "and"), "a and b");
  EXPECT_EQ(util::grammatically_join({"a", "b"}, "and", false), "a and b");
  EXPECT_EQ(util::grammatically_join({"a", "b", "c"}, "or"), "a, b, or c");
  EXPECT_EQ(util::grammatically_join({"a", "b", "c"}, "or", false), "a, b or c");
}

TEST(StringUtil, parse_bytes) {
  EXPECT_EQ(util::parse_bytes("256MiB"), 256ull << 20);
  EXPECT_EQ(util::parse_bytes("256MB"), 256ull * 1000 * 1000);
  EXPECT_EQ(util::parse_bytes("1GiB"), 1ull << 30);
  EXPECT_EQ(util::parse_bytes("1.5GiB"), 1.5 * (1ull << 30));
  EXPECT_EQ(util::parse_bytes("1073741824"), 1073741824ull);
}

TEST(StringUtil, float_to_str8) {
  // blank_zeros=true (default): zero returns empty string
  EXPECT_EQ(util::float_to_str8(0.0f, true), "");
  EXPECT_EQ(util::float_to_str8(0.0f), "");

  // blank_zeros=false: zero returns a non-empty string of length <= 8
  std::string zero_str = util::float_to_str8(0.0f, false);
  EXPECT_FALSE(zero_str.empty());
  EXPECT_LE(zero_str.size(), 8u);

  // Normal floats: result fits in 8 chars
  EXPECT_EQ(util::float_to_str8(1.5f), "1.5");
  EXPECT_EQ(util::float_to_str8(3.0f), "3");
  EXPECT_EQ(util::float_to_str8(-1.5f), "-1.5");

  // All outputs must fit in 8 chars
  for (float v : {0.1f, 0.5f, 1.0f, 1.23456789f, 100.0f, -42.5f, 1e-4f, 1e6f}) {
    std::string s = util::float_to_str8(v);
    EXPECT_LE(s.size(), 8u) << "float_to_str8(" << v << ") = \"" << s << "\" (len " << s.size() << ")";
  }
}

TEST(StringUtil, atoi) {
  EXPECT_EQ(util::atoi("0"), 0);
  EXPECT_EQ(util::atoi("1"), 1);
  EXPECT_EQ(util::atoi("42"), 42);
  EXPECT_EQ(util::atoi("-7"), -7);
  EXPECT_EQ(util::atoi("-123"), -123);
  EXPECT_EQ(util::atoi("2147483647"), 2147483647);  // INT_MAX

  EXPECT_THROW(util::atoi(""), std::invalid_argument);
  EXPECT_THROW(util::atoi("abc"), std::invalid_argument);
  EXPECT_THROW(util::atoi("1.5"), std::invalid_argument);
  EXPECT_THROW(util::atoi("1a"), std::invalid_argument);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
