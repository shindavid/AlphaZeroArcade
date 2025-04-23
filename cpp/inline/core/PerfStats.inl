#include <core/PerfStats.hpp>

#include <util/Asserts.hpp>
#include <util/CppUtil.hpp>

namespace core {


inline SearchThreadPerfStats& SearchThreadPerfStats::operator+=(
  const SearchThreadPerfStats& other) {
  cache_hits += other.cache_hits;
  cache_misses += other.cache_misses;

  wait_for_game_slot_time_ns += other.wait_for_game_slot_time_ns;
  cache_mutex_acquire_time_ns += other.cache_mutex_acquire_time_ns;
  cache_insert_time_ns += other.cache_insert_time_ns;
  batch_prepare_time_ns += other.batch_prepare_time_ns;
  batch_write_time_ns += other.batch_write_time_ns;
  wait_for_nn_eval_time_ns += other.wait_for_nn_eval_time_ns;
  mcts_time_ns += other.mcts_time_ns;
  return *this;
}

inline void SearchThreadPerfStats::fill_json(boost::json::object& obj) const {
  obj["cache_hits"] = cache_hits;
  obj["cache_misses"] = cache_misses;

  obj["wait_for_game_slot_time_ns"] = wait_for_game_slot_time_ns;
  obj["cache_mutex_acquire_time_ns"] = cache_mutex_acquire_time_ns;
  obj["cache_insert_time_ns"] = cache_insert_time_ns;
  obj["batch_prepare_time_ns"] = batch_prepare_time_ns;
  obj["batch_write_time_ns"] = batch_write_time_ns;
  obj["wait_for_nn_eval_time_ns"] = wait_for_nn_eval_time_ns;
  obj["mcts_time_ns"] = mcts_time_ns;
}

inline void SearchThreadPerfStats::normalize(int num_game_threads) {
  wait_for_game_slot_time_ns /= num_game_threads;
  cache_mutex_acquire_time_ns /= num_game_threads;
  cache_insert_time_ns /= num_game_threads;
  batch_prepare_time_ns /= num_game_threads;
  batch_write_time_ns /= num_game_threads;
  wait_for_nn_eval_time_ns /= num_game_threads;
  mcts_time_ns /= num_game_threads;
}

inline NNEvalLoopPerfStats& NNEvalLoopPerfStats::operator+=(
  const NNEvalLoopPerfStats& other) {
  positions_evaluated += other.positions_evaluated;
  batches_evaluated += other.batches_evaluated;
  full_batches_evaluated += other.full_batches_evaluated;

  wait_for_search_threads_time_ns += other.wait_for_search_threads_time_ns;
  cpu2gpu_copy_time_ns += other.cpu2gpu_copy_time_ns;
  gpu2cpu_copy_time_ns += other.gpu2cpu_copy_time_ns;
  model_eval_time_ns += other.model_eval_time_ns;

  batch_datas_allocated += other.batch_datas_allocated;
  return *this;
}

inline void NNEvalLoopPerfStats::fill_json(boost::json::object& obj) const {
  obj["positions_evaluated"] = positions_evaluated;
  obj["batches_evaluated"] = batches_evaluated;
  obj["full_batches_evaluated"] = full_batches_evaluated;

  obj["wait_for_search_threads_time_ns"] = wait_for_search_threads_time_ns;
  obj["cpu2gpu_copy_time_ns"] = cpu2gpu_copy_time_ns;
  obj["gpu2cpu_copy_time_ns"] = gpu2cpu_copy_time_ns;
  obj["model_eval_time_ns"] = model_eval_time_ns;

  obj["batch_datas_allocated"] = batch_datas_allocated;
}

inline LoopControllerPerfStats& LoopControllerPerfStats::operator+=(
  const LoopControllerPerfStats& other) {
  pause_time_ns += other.pause_time_ns;
  model_load_time_ns += other.model_load_time_ns;
  total_time_ns += other.total_time_ns;

  return *this;
}

inline void LoopControllerPerfStats::fill_json(boost::json::object& obj) const {
  obj["pause_time_ns"] = pause_time_ns;
  obj["model_load_time_ns"] = model_load_time_ns;
  obj["total_time_ns"] = total_time_ns;
}

inline PerfStats& PerfStats::operator+=(const PerfStats& other) {
  search_thread_stats += other.search_thread_stats;
  nn_eval_loop_stats += other.nn_eval_loop_stats;
  loop_controller_stats += other.loop_controller_stats;
  return *this;
}

inline boost::json::object PerfStats::to_json() const {
  boost::json::object obj;
  search_thread_stats.fill_json(obj);
  nn_eval_loop_stats.fill_json(obj);
  loop_controller_stats.fill_json(obj);
  return obj;
}

inline void PerfStats::update(const SearchThreadPerfStats& stats) {
  search_thread_stats += stats;
}

inline void PerfStats::update(const NNEvalLoopPerfStats& stats) {
  nn_eval_loop_stats += stats;
}

inline void PerfStats::update(const LoopControllerPerfStats& stats) {
  loop_controller_stats += stats;
}

inline void PerfStats::calibrate(int num_game_threads) {
  search_thread_stats.normalize(num_game_threads);

  // mcts-time-ns includes everything except for wait-for-game-slot-time-ns. Let's undo that.
  search_thread_stats.mcts_time_ns -= search_thread_stats.cache_mutex_acquire_time_ns;
  search_thread_stats.mcts_time_ns -= search_thread_stats.cache_insert_time_ns;
  search_thread_stats.mcts_time_ns -= search_thread_stats.batch_prepare_time_ns;
  search_thread_stats.mcts_time_ns -= search_thread_stats.batch_write_time_ns;
  search_thread_stats.mcts_time_ns -= search_thread_stats.wait_for_nn_eval_time_ns;
  if (search_thread_stats.mcts_time_ns < 0) {
    search_thread_stats.mcts_time_ns = 0;
  }

  // pause time includes reload time. Let's undo that.
  util::release_assert(
    loop_controller_stats.pause_time_ns >= loop_controller_stats.model_load_time_ns,
    "pause_time_ns < model_load_time_ns (%ld < %ld)", loop_controller_stats.pause_time_ns,
    loop_controller_stats.model_load_time_ns);
  loop_controller_stats.pause_time_ns -= loop_controller_stats.model_load_time_ns;
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
  field_ += util::to_ns(stop_time_ - start_time_);
}

}  // namespace core
