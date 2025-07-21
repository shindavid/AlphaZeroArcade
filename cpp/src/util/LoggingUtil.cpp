#include "util/LoggingUtil.hpp"

#include <spdlog/common.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdarg>
#include <ctime>
#include <vector>

namespace util {

int Logging::kTimestampPrefixLength;

// TODO: Change logging to use nanosecond precision
void Logging::init(const Params& params) {
  // Collect sinks
  std::vector<spdlog::sink_ptr> sinks;

  const char* format = params.omit_timestamps ? "%v" : "%Y-%m-%d %H:%M:%S.%f %v";

  if (params.omit_timestamps) {
    kTimestampPrefixLength = 0;
  } else {
    kTimestampPrefixLength = 27;  // "2024-03-12 17:13:11.259615 "
  }

  // Console sink
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_pattern(format);
  sinks.push_back(console_sink);

  // File sink, if needed
  if (!params.log_filename.empty()) {
    auto file_sink =
      std::make_shared<spdlog::sinks::basic_file_sink_mt>(params.log_filename, !params.append_mode);
    file_sink->set_pattern(format);
    sinks.push_back(file_sink);
  }

  // Create and set the default logger
  auto logger = std::make_shared<spdlog::logger>("multi_sink", sinks.begin(), sinks.end());
  spdlog::set_default_logger(logger);

  // Now set up flushing behavior on the logger itself
  spdlog::flush_on(spdlog::level::debug);

  // enable all levels of logging (filtering is done at compile-level)
  spdlog::set_level(spdlog::level::trace);
}

}  // namespace util
