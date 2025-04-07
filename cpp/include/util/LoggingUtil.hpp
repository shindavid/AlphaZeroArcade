#pragma once

#include <spdlog/fmt/ostr.h>  // Enables fallback to ostream <<
#include <spdlog/spdlog.h>

#include <string>

#define LOG_INFO SPDLOG_INFO
#define LOG_DEBUG SPDLOG_DEBUG
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
