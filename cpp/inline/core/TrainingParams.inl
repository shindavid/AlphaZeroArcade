#include "core/TrainingParams.hpp"

#include "util/BoostUtil.hpp"

#include <boost/program_options.hpp>

namespace core {

inline TrainingParams& TrainingParams::instance() {
  static TrainingParams params;
  return params;
}

inline auto TrainingParams::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("TrainingDataWriter options");
  return desc
    .template add_option<"max-rows", 'M'>(po::value<int64_t>(&max_rows)->default_value(max_rows),
                                          "if specified, kill process after writing this many rows")
    .template add_option<"heartbeat-frequency-seconds", 'H'>(
      po::value<float>(&heartbeat_frequency_seconds)->default_value(heartbeat_frequency_seconds),
      "heartbeat frequency in seconds")
    .template add_flag<"enable-training", "disable-training">(&enabled, "enable training",
                                                              "disable training");
}

}  // namespace core
