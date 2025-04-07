#pragma once

#include <spdlog/fmt/ostr.h>  // Enables fallback to ostream <<
#include <spdlog/spdlog.h>

#include <string>

// The main logging macros are LOG_INFO(), LOG_DEBUG(), LOG_WARN(), and LOG_ERROR().
//
// These use std::format() to format the message. For example:
//
// LOG_INFO("Hello {}!", "world");
// LOG_DEBUG("x={} pi={}", 3, 3.14159);
//
// By default, LOG_DEBUG() statements are compiled out. In order to enable them, pass
// --enable-debug-logging to py/build.py

#define LOG_DEBUG SPDLOG_DEBUG
#define LOG_INFO SPDLOG_INFO
#define LOG_WARN SPDLOG_WARN
#define LOG_ERROR SPDLOG_ERROR

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

#include <inline/util/LoggingUtil.inl>
