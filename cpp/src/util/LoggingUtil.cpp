#include <util/LoggingUtil.hpp>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>

#include <chrono>
#include <cstdarg>
#include <ctime>
#include <iostream>

namespace util {

int Logging::kTimestampPrefixLength;

// TODO: Change logging to use nanosecond precision
void Logging::init(const Params& params) {
  namespace trivial = boost::log::trivial;
  namespace keywords = boost::log::keywords;

  const char* format = "%TimeStamp% %Message%";
  if (params.omit_timestamps) {
    format = "%Message%";
    kTimestampPrefixLength = 0;
  } else {
    kTimestampPrefixLength = 27;  // "2024-03-12 17:13:11.259615 "
  }

  boost::log::add_console_log(std::cout, keywords::auto_flush = true, keywords::format = format);

  if (!params.log_filename.empty()) {
    auto open_mode = params.append_mode ? std::ios_base::app : std::ios_base::out;
    boost::log::add_file_log(keywords::file_name = params.log_filename.c_str(),
                             keywords::auto_flush = true, keywords::open_mode = open_mode,
                             keywords::format = format);
  }
  if (params.debug) {
    boost::log::core::get()->set_filter(trivial::severity >= trivial::debug);
  }
  boost::log::add_common_attributes();
}

}  // namespace util
