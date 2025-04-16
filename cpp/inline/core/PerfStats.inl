#include <core/PerfStats.hpp>

namespace core {

inline PerfStats& PerfStats::operator+=(const PerfStats& other) {
  cache_hits += other.cache_hits;
  cache_misses += other.cache_misses;
  positions_evaluated += other.positions_evaluated;
  batches_evaluated += other.batches_evaluated;
  full_batches_evaluated += other.full_batches_evaluated;

  check_cache_mutex_time_ns += other.check_cache_mutex_time_ns;
  check_cache_insert_time_ns += other.check_cache_insert_time_ns;
  check_cache_alloc_time_ns += other.check_cache_alloc_time_ns;
  check_cache_set_time_ns += other.check_cache_set_time_ns;
  batch_ready_wait_time_ns += other.batch_ready_wait_time_ns;
  gpu_copy_time_ns += other.gpu_copy_time_ns;
  model_eval_time_ns += other.model_eval_time_ns;

  batch_datas_allocated += other.batch_datas_allocated;
  return *this;
}

inline boost::json::object PerfStats::to_json() const {
  boost::json::object obj;
  obj["cache_hits"] = cache_hits;
  obj["cache_misses"] = cache_misses;
  obj["positions_evaluated"] = positions_evaluated;
  obj["batches_evaluated"] = batches_evaluated;
  obj["full_batches_evaluated"] = full_batches_evaluated;

  obj["check_cache_mutex_time_ns"] = check_cache_mutex_time_ns;
  obj["check_cache_insert_time_ns"] = check_cache_insert_time_ns;
  obj["check_cache_alloc_time_ns"] = check_cache_alloc_time_ns;
  obj["check_cache_set_time_ns"] = check_cache_set_time_ns;
  obj["batch_ready_wait_time_ns"] = batch_ready_wait_time_ns;
  obj["gpu_copy_time_ns"] = gpu_copy_time_ns;
  obj["model_eval_time_ns"] = model_eval_time_ns;

  obj["batch_datas_allocated"] = batch_datas_allocated;
  return obj;
}

inline PerfStatsClocker::PerfStatsClocker(int64_t& field) : field_(field) {
  start_time_ = std::chrono::steady_clock::now();
}

// Use this to simultaneously start a new PerfStatsClocker and stop the previous one.
inline PerfStatsClocker::PerfStatsClocker(PerfStatsClocker& previous, int64_t& field)
    : field_(field) {
  previous.stop();
  start_time_ = previous.stop_time_;
}

// Call this to manually stop the timer, without waiting for the destructor to do it.
inline void PerfStatsClocker::stop() {
  if (stopped_) return;
  stopped_ = true;
  stop_time_ = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time_ - start_time_);

  field_ += duration.count();
}

}  // namespace core
