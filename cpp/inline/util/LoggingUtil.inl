#include <util/LoggingUtil.hpp>

#include <util/BoostUtil.hpp>

#include <boost/program_options.hpp>

#include <chrono>
#include <ctime>
#include <iostream>

namespace util {

namespace logging {

inline auto Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("Logging options");

  return desc
      .template add_option<"log-filename">(
          po::value<std::string>(&log_filename),
          "log filename. If specified, logs to the file in addition to stdout")
      .template add_flag<"debug", "no-debug">(&debug, "enable debug logging",
                                              "disable debug logging");
}

}  // namespace logging

}  // namespace util
