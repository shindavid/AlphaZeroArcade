#pragma once

#include <boost/json.hpp>

#include <cstdint>

/*
 * Each mcts::NNEvaluationService keeps track of its own performance statistics, in the form of
 * a PerfStats object. In theory there can be multiple NNEvaluationService's running in a
 * single process, so we aggregate their stats for reporting purposes.
 */
namespace core {

struct PerfStats {
  PerfStats& operator+=(const PerfStats& other);
  boost::json::object to_json() const;
  bool empty() const { return cache_hits + cache_misses == 0; }

  int64_t cache_hits = 0;
  int64_t cache_misses = 0;
  int64_t positions_evaluated = 0;
  int64_t batches_evaluated = 0;
  int64_t full_batches_evaluated = 0;

  int64_t check_cache_mutex_time_ns = 0;
  int64_t check_cache_insert_time_ns = 0;
  int64_t check_cache_alloc_time_ns = 0;
  int64_t check_cache_set_time_ns = 0;
  int64_t batch_ready_wait_time_ns = 0;
  int64_t gpu_copy_time_ns = 0;
  int64_t model_eval_time_ns = 0;

  int batch_datas_allocated = 0;
};

// PerfStatsClocker can be used to measure the time taken for a specific operation. Its constructor
// starts the timer, and its destructor stops the timer and updates a specific passed-in field,
// which is expected to be a reference to a PerfStats int64_t that represents cumulative time in
// nanoseconds.
class PerfStatsClocker {
 public:
  // Standard constructor: pass in a reference to the nanosecond field to update.
  PerfStatsClocker(int64_t& field);

  // Use this to simultaneously start a new PerfStatsClocker and stop a previous one.
  PerfStatsClocker(PerfStatsClocker& previous, int64_t& field);

  ~PerfStatsClocker() { stop(); }

  // Call this to manually stop the timer, without waiting for the destructor to do it.
  void stop();

 private:
  int64_t& field_;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point stop_time_;
  bool stopped_ = false;
};

}  // namespace core

#include <inline/core/PerfStats.inl>
