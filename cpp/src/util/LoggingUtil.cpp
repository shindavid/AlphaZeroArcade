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

namespace logging {

// TODO: Change logging to use nanosecond precision
void init(const Params& params) {
  namespace trivial = boost::log::trivial;
  namespace keywords = boost::log::keywords;

  if (!params.log_filename.empty()) {
    auto f = params.log_filename.c_str();
    boost::log::add_file_log(f, keywords::auto_flush = true,
                             keywords::format = "%TimeStamp% %Message%");
  } else {
    boost::log::add_console_log(std::cout, keywords::format = "%TimeStamp% %Message%");
  }
  if (params.debug) {
    boost::log::core::get()->set_filter(trivial::severity >= trivial::debug);
  }
  boost::log::add_common_attributes();
}

}  // namespace logging

}  // namespace util
