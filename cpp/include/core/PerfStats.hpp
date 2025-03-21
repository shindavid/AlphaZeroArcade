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
};

}  // namespace core

#include <inline/core/PerfStats.inl>
