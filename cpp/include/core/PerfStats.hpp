#pragma once

#include <boost/json.hpp>

#include <cstdint>

/*
 * Each mcts::NNEvaluationService keeps track of its own performance statistics, in the form of
 * a PerfStats object. In theory there can be multiple NNEvaluationService's running in a
 * single process, so we aggregate their stats for reporting purposes.
 */
namespace core {

// Component of PerfStats that tracks performance from the perspective of the search threads.
struct SearchThreadPerfStats {
  SearchThreadPerfStats& operator+=(const SearchThreadPerfStats& other);
  void fill_json(boost::json::object& obj) const;
  void normalize(int num_game_threads);

  int64_t cache_hits = 0;
  int64_t cache_misses = 0;

  int64_t wait_for_game_slot_time_ns = 0;
  int64_t cache_mutex_acquire_time_ns = 0;
  int64_t cache_insert_time_ns = 0;
  int64_t batch_prepare_time_ns = 0;
  int64_t batch_write_time_ns = 0;
  int64_t wait_for_nn_eval_time_ns = 0;
  int64_t mcts_time_ns = 0;
};

// Component of PerfStats that tracks performance from the perspective of the NNEvaluationService
// schedule loop.
struct NNEvalScheduleLoopPerfStats {
  NNEvalScheduleLoopPerfStats& operator+=(const NNEvalScheduleLoopPerfStats& other);
  void fill_json(boost::json::object& obj) const;

  int64_t positions_evaluated = 0;
  int64_t batches_evaluated = 0;
  int64_t full_batches_evaluated = 0;

  int64_t wait_for_search_threads_time_ns = 0;
  int64_t pipeline_wait_time_ns = 0;
  int64_t pipeline_schedule_time_ns = 0;

  int batch_datas_allocated = 0;
};

struct LoopControllerPerfStats {
  LoopControllerPerfStats& operator+=(const LoopControllerPerfStats& other);
  void fill_json(boost::json::object& obj) const;

  int64_t pause_time_ns = 0;
  int64_t model_load_time_ns = 0;
  int64_t total_time_ns = 0;
};

struct PerfStats {
  PerfStats& operator+=(const PerfStats& other);
  boost::json::object to_json() const;
  void update(const SearchThreadPerfStats&);
  void update(const NNEvalScheduleLoopPerfStats&);
  void update(const LoopControllerPerfStats&);
  void calibrate(int num_game_threads);

  SearchThreadPerfStats search_thread_stats;
  NNEvalScheduleLoopPerfStats nn_eval_schedule_loop_stats;
  LoopControllerPerfStats loop_controller_stats;
};

// A PerfStatsClient is an object that profiles its own performance.
class PerfStatsClient {
 public:
  // Constructor registers the client with the PerfStatsRegistry.
  PerfStatsClient();

  virtual ~PerfStatsClient() = default;

  // Add my stats to the passed-in PerfStats object. Should clear my own stats after updating stats.
  virtual void update_perf_stats(PerfStats& stats) = 0;
};

class PerfStatsRegistry {
 public:
  static PerfStatsRegistry* instance();
  static void clear() { instance()->clients_.clear(); }
  void add(PerfStatsClient* client);

  PerfStats get_perf_stats();

 private:
  std::vector<PerfStatsClient*> clients_;
};

// PerfClocker can be used to measure the time taken for a specific operation. Its constructor
// starts the timer, and its destructor stops the timer and updates a specific passed-in field,
// which is expected to be a reference to a PerfStats int64_t that represents cumulative time in
// nanoseconds.
class PerfClocker {
 public:
  // Standard constructor: pass in a reference to the nanosecond field to update.
  PerfClocker(int64_t& field);

  // Use this to simultaneously start a new PerfClocker and stop a previous one.
  PerfClocker(PerfClocker& previous, int64_t& field);

  ~PerfClocker() { stop(); }

  // Call this to manually stop the timer, without waiting for the destructor to do it.
  void stop();

 private:
  int64_t& field_;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point stop_time_;
  bool stopped_ = false;
};

}  // namespace core

#include "inline/core/PerfStats.inl"
