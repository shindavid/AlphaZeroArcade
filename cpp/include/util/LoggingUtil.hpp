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

namespace logging {

struct Params {
  std::string log_filename;
  bool debug = false;

  auto make_options_description();
};

void init(const Params&);

constexpr int kTimestampPrefixLength = 27;  // "2024-03-12 17:13:11.259615 "

}  // namespace logging

}  // namespace util

#include <inline/util/LoggingUtil.inl>
