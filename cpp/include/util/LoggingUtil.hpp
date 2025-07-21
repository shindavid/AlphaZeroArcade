#pragma once

#include <spdlog/fmt/ostr.h>  // Enables fallback to ostream <<
#include <spdlog/spdlog.h>

#include <string>
#include "util/CppUtil.hpp"

// The main logging macros are LOG_INFO(), LOG_DEBUG(), LOG_WARN(), and LOG_ERROR().
//
// These use std::format() to format the message. For example:
//
// LOG_INFO("Hello {}!", "world");
// LOG_DEBUG("x={} pi={}", 3, 3.14159);
//
// By default, LOG_DEBUG() statements are compiled out. In order to enable them, pass
// --enable-debug-logging to py/build.py

#define LOG_TRACE(...)            \
  do {                            \
    USE_UNEVALUATED(__VA_ARGS__); \
    SPDLOG_TRACE(__VA_ARGS__);    \
  } while (0)

#define LOG_DEBUG(...)            \
  do {                            \
    USE_UNEVALUATED(__VA_ARGS__); \
    SPDLOG_DEBUG(__VA_ARGS__);    \
  } while (0)

#define LOG_INFO(...)             \
  do {                            \
    USE_UNEVALUATED(__VA_ARGS__); \
    SPDLOG_INFO(__VA_ARGS__);     \
  } while (0)

#define LOG_WARN(...)             \
  do {                            \
    USE_UNEVALUATED(__VA_ARGS__); \
    SPDLOG_WARN(__VA_ARGS__);     \
  } while (0)

#define LOG_ERROR(...)            \
  do {                            \
    USE_UNEVALUATED(__VA_ARGS__); \
    SPDLOG_ERROR(__VA_ARGS__);    \
  } while (0)

namespace util {

struct Logging {
  struct Params {
    std::string log_filename;
    bool append_mode = false;
    bool omit_timestamps = false;

    auto make_options_description();
  };

  static void init(const Params&);

  static int kTimestampPrefixLength;
};  // Logging

}  // namespace util

#include "inline/util/LoggingUtil.inl"
