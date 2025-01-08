#pragma once

#include <iostream>
#include <mutex>
#include <string>

#include <boost/log/trivial.hpp>

#define LOG_INFO BOOST_LOG_TRIVIAL(info)
#define LOG_DEBUG BOOST_LOG_TRIVIAL(debug)
#define LOG_WARN BOOST_LOG_TRIVIAL(warning)
#define LOG_ERROR BOOST_LOG_TRIVIAL(error)

namespace util {

struct Logging {

struct Params {
  std::string log_filename;
  bool debug = false;
  bool omit_timestamps = false;

  auto make_options_description();
};

static void init(const Params&);

static int kTimestampPrefixLength;

};  // Logging

}  // namespace util

#include <inline/util/LoggingUtil.inl>
