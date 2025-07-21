#include "util/LoggingUtil.hpp"

#include "util/BoostUtil.hpp"

#include <boost/program_options.hpp>

#include <ctime>

namespace util {

inline auto Logging::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("Logging options");

  return desc
    .template add_option<"log-filename">(
      po::value<std::string>(&log_filename),
      "log filename. If specified, logs to the file in addition to stdout")
    .template add_flag<"log-append-mode", "log-write-mode">(
      &append_mode, "write log in append mode", "write log in write mode")
    .template add_flag<"omit-timestamps", "include-timestamps">(&omit_timestamps, "omit timestamps",
                                                                "include timestamps");
}

}  // namespace util
