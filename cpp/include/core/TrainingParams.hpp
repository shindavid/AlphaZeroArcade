#pragma once

#include <cstdint>

namespace core {

// TrainingParams is used by core::TrainingDataWriter.
struct TrainingParams {
  static TrainingParams& instance();
  auto make_options_description();

  int64_t max_rows = 0;
  float heartbeat_frequency_seconds = 1.0;
  bool enabled = false;

  // Fields set separately from the command-line options:
  int num_game_threads = 0;
};

}  // namespace core

#include "inline/core/TrainingParams.inl"
